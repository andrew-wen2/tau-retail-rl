"""Episode record format for the GRPO loop: schema, writer, reader, and the
prefix-consistency gate.

This file is the interface between the rollout driver (.venv-tau2, on the VM) and the
trainer (.venv-train). It is the one artifact the design calls expensive to change, so it
carries a schema_version and nothing else in the loop may write episodes another way.

Three facts drive the design:

  * The driver and the trainer live in different venvs and, during development, on different
    machines. So the format is stdlib-only -- gzipped JSON, no numpy, no torch. The trainer
    converts to tensors at read time.

  * Records are immutable and written once with an atomic rename. Anything that might be
    recomputed later under a changed rule is DERIVED at read time rather than stored:
    the training-inclusion rule (from `termination`) and the
    authentication-before-first-write flag both work this way. The authentication check is
    a scan of trajectories the loop already produces, so
    storing the flag would freeze a rule that should stay re-appliable to old records.

  * Requirement 2 ("track tokens, never re-render") cannot be satisfied by capture alone.
    vLLM re-templates the whole history on every chat request, so what the reader can do is
    TEST whether turn k's prompt survived into turn k+1 unchanged. That test is the
    prefix-consistency gate, and it decides whether an episode trains as one ~12k sequence
    or as N per-turn sequences at ~8x the compute. See gate_report().

Storage ceiling, stated so the upgrade trigger is explicit: one gzipped JSON file per
episode is roughly 100KB at ~99k prompt tokens, so ~50MB per 500-episode step and ~2.5GB
across a 50-step run. That fits the 150GB disk. If JSON parse time becomes the trainer's
bottleneck, move `prompt_token_ids` to a binary sidecar addressed by (offset, length) --
the prefix property makes that encoding nearly free, but do not build it before the gate
says the property holds.

CLI:
    python3 records.py gate <dir>       prefix-consistency gate over a record dir
    python3 records.py spans <dir>      token diffs for episodes that failed the gate
    python3 records.py inspect <file>   human-readable dump of one episode
    python3 records.py profile <dir>    per-task {successes, trials} over trainable episodes
    python3 records.py selftest         round-trip + gate math, no data needed
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1

# The gate threshold from the design: at or above this, single-sequence assembly is the
# design and per-turn stays a rare safety valve. Below it, stop and build construction
# (/v1/completions with token-ID prompts) rather than absorbing an ~8x compute increase.
GATE_THRESHOLD = 0.95

# Retail tools that only read. Kept in sync with convert_tb.py, which defines the
# same set for a different purpose (explaining a gold-replay result). If retail's tool list
# ever changes, both move together.
READ_ONLY_TOOLS = frozenset({
    "calculate", "find_user_id_by_email", "find_user_id_by_name_zip", "get_item_details",
    "get_order_details", "get_product_details", "get_user_details",
    "list_all_product_types", "think", "transfer_to_human_agents",
})  # get_item_details was missing until Sept 26 (tau2 marks it ToolType.READ): the auth scan
    # treated it as a write, so episodes that looked up an item before authenticating counted
    # as writing unauthenticated

# Authentication check: the agent is expected to identify the customer before it mutates anything.
AUTH_TOOLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})

# Termination classes and whether the episode contributes gradient. Derived, not stored.
#   natural / max_turns / empty_message / tool_parse_error / invalid_tool_args /
#   too_many_errors
#       -> policy failures, they train (at reward 0 where the env gave none).
#          max_turns is tau2's max_steps (200 messages); too_many_errors is its max_errors
#          (10 failed tool calls), both mirrored so training ends episodes where eval does.
#   no_progress_timeout
#       -> completed turns train at reward 0; the in-flight turn returned nothing, because
#          requests are non-streaming, so there is nothing to drop
#   length
#       -> trains at reward 0. Originally excluded ("a truncated turn teaches truncation"),
#          which is backwards: at reward 0 its advantage is negative and pushes AWAY from
#          truncation, while exclusion made truncation free. The first integration run found
#          exactly that (Sept 23): by step 8 the policy had turned nano goodbye loops, which
#          score 0, into one unbounded farewell turn that hit the 2048-token cap and dropped
#          out of the gradient (length 0 -> 42 of 128 episodes).
#   customer_api_error / env_exception
#       -> infrastructure, not policy. Excluded and re-rolled in-batch.
TERMINATION_CLASSES: dict[str, bool] = {
    "natural": True,
    "no_progress_timeout": True,
    "max_turns": True,
    "empty_message": True,
    "tool_parse_error": True,
    "invalid_tool_args": True,
    "too_many_errors": True,
    "length": True,
    "customer_api_error": False,
    "env_exception": False,
}

RE_ROLL_CLASSES = frozenset({"customer_api_error", "env_exception"})


class SchemaError(ValueError):
    """A record does not match SCHEMA_VERSION or is internally inconsistent."""


@dataclass
class AssistantTurn:
    """One policy generation. prompt_token_ids and token_ids are exactly what vLLM reported
    via return_token_ids -- never re-tokenized text, never a re-rendered template."""

    prompt_token_ids: list[int]
    token_ids: list[int]
    logprobs: list[float]  # sampled-token logprob per generated token; TIS needs only these
    text: str  # what the server returned, kept for transcript reading, never trained on
    finish_reason: str
    tool_call: dict[str, Any] | None = None  # {"name": str, "arguments": dict}; the first call
    role: str = "assistant"
    # Every call in the turn, in order, as {"id", "name", "arguments"}. tau2 executes all of
    # them, so the driver does too; `tool_call` stays the first one for older readers.
    tool_calls: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if len(self.logprobs) != len(self.token_ids):
            raise SchemaError(
                f"logprobs/token_ids length mismatch: "
                f"{len(self.logprobs)} vs {len(self.token_ids)}"
            )


@dataclass
class EnvTurn:
    """A customer message or a tool result, as it was appended to the history. Unmasked and
    untrained; present so an episode can be read as a transcript and replayed offline."""

    role: str  # "user" (the nano customer) or "tool"
    content: str
    tool_name: str | None = None


@dataclass
class Episode:
    episode_id: str
    batch_id: int
    policy_version: str  # "{step:03d}-{sha12}", or "base-b7ea907" for the recording pass
    task_id: str
    group_id: str
    trial_index: int
    seed: int
    turns: list[AssistantTurn | EnvTurn]
    reward: int
    termination: str
    db_hash_before: str
    db_hash_after: str
    wall_clock_s: float
    recorded_at: str
    schema_version: int = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.termination not in TERMINATION_CLASSES:
            raise SchemaError(
                f"unknown termination class {self.termination!r}; "
                f"known: {sorted(TERMINATION_CLASSES)}"
            )
        if self.reward not in (0, 1):
            raise SchemaError(f"reward must be 0 or 1, got {self.reward!r}")

    @property
    def assistant_turns(self) -> list[AssistantTurn]:
        return [t for t in self.turns if isinstance(t, AssistantTurn)]

    @property
    def turn_count(self) -> int:
        return len(self.assistant_turns)

    @property
    def eval_reward(self) -> int:
        """The reward tau2's evaluator would give: 0 unless the conversation ended naturally.
        Equal to `reward` for records written after Sept 23; applied at read time so records
        written before the driver mirrored the rule are scored the same way."""
        return self.reward if self.termination == "natural" else 0

    @property
    def trains(self) -> bool:
        """Whether this episode contributes gradient. Derived from termination, not stored,
        so the rule can change without invalidating recorded episodes."""
        return TERMINATION_CLASSES[self.termination]

    @property
    def needs_reroll(self) -> bool:
        """Infrastructure failure: re-roll in-batch, bounded by the drain deadline, so
        nothing spans a policy version. A group that cannot be refilled is dropped whole
        rather than shipped short -- the sqrt(G-1) advantage bound assumes full G."""
        return self.termination in RE_ROLL_CLASSES


# --------------------------------------------------------------------------- serialization


def _to_jsonable(ep: Episode) -> dict[str, Any]:
    d = asdict(ep)
    d["turns"] = [
        {"_k": "a", **asdict(t)} if isinstance(t, AssistantTurn) else {"_k": "e", **asdict(t)}
        for t in ep.turns
    ]
    return d


def _from_jsonable(d: dict[str, Any]) -> Episode:
    got = d.get("schema_version")
    if got != SCHEMA_VERSION:
        raise SchemaError(
            f"schema_version {got!r} != {SCHEMA_VERSION}. "
            "Records are immutable; write a migration rather than reinterpreting in place."
        )
    turns: list[AssistantTurn | EnvTurn] = []
    for t in d["turns"]:
        kind = t.pop("_k")
        turns.append(AssistantTurn(**t) if kind == "a" else EnvTurn(**t))
    d = dict(d, turns=turns)
    return Episode(**d)


def write_episode(ep: Episode, out_dir: str | Path) -> Path:
    """Write one episode atomically. Records are written once and never mutated, so a torn
    write must not be readable: serialize to a temp file in the same directory, fsync, then
    rename. A crash mid-write leaves the temp file, which the reader ignores."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{ep.episode_id}.json.gz"
    payload = json.dumps(_to_jsonable(ep), separators=(",", ":")).encode()

    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".tmp-", suffix=".json.gz")
    try:
        with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(payload)
            gz.flush()
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, final)
        # fsync the directory too: without it a hard stop (Spot preemption) can persist the
        # rename but not the data, which left one truncated record on Sept 23.
        dfd = os.open(out_dir, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return final


def read_episode(path: str | Path) -> Episode:
    with gzip.open(path, "rb") as fh:
        return _from_jsonable(json.loads(fh.read()))


def iter_episodes(rec_dir: str | Path) -> Iterator[Episode]:
    """Yield episodes in stable filename order. Temp files from torn writes are skipped."""
    for p in sorted(Path(rec_dir).glob("*.json.gz")):
        if p.name.startswith(".tmp-"):
            continue
        yield read_episode(p)


# ------------------------------------------------------------------ prefix-consistency gate


@dataclass
class TurnCheck:
    index: int
    prefix_ok: bool
    mask_ok: bool

    @property
    def ok(self) -> bool:
        return self.prefix_ok and self.mask_ok


def check_episode(ep: Episode) -> list[TurnCheck]:
    """Both checks from the design, for every adjacent pair of assistant turns.

      1. prompt_k is a token-exact prefix of prompt_{k+1}
      2. prompt_{k+1}[len(prompt_k) : len(prompt_k)+len(out_k)] == out_k

    The second is not optional and is not implied by the first. Template glue (<|im_end|>,
    role headers) can shift where turn k's completion actually sits, and a mask placed on
    the wrong offset trains the wrong tokens with no error anywhere.
    """
    turns = ep.assistant_turns
    checks: list[TurnCheck] = []
    for k in range(len(turns) - 1):
        cur, nxt = turns[k], turns[k + 1]
        n = len(cur.prompt_token_ids)
        prefix_ok = nxt.prompt_token_ids[:n] == cur.prompt_token_ids
        span = nxt.prompt_token_ids[n : n + len(cur.token_ids)]
        mask_ok = prefix_ok and span == cur.token_ids
        checks.append(TurnCheck(index=k, prefix_ok=prefix_ok, mask_ok=mask_ok))
    return checks


def episode_is_single_sequence(ep: Episode) -> bool:
    """True when the whole episode can train as one sequence. A single-turn episode is
    trivially assemblable: there is no adjacent pair that could have been re-rendered."""
    return all(c.ok for c in check_episode(ep))


@dataclass
class Assembled:
    """One training example. `token_ids` is the sequence; `loss_mask[i]` is True exactly
    where the policy generated token i. Advantage is broadcast across the masked tokens by
    the trainer, which owns the GRPO math."""

    token_ids: list[int]
    loss_mask: list[bool]
    logprobs: list[float | None]  # behavior logprob where masked, None elsewhere

    def __post_init__(self) -> None:
        if not (len(self.token_ids) == len(self.loss_mask) == len(self.logprobs)):
            raise SchemaError("assembled arrays must be the same length")

    @property
    def n_trained_tokens(self) -> int:
        return sum(self.loss_mask)


def assemble(ep: Episode) -> list[Assembled]:
    """Single sequence when the checks pass, per-turn otherwise.

    Per-turn is correct but roughly 8x the compute, so it is a safety valve and not a mode.
    It is also the ONLY correct assembly where re-rendering occurred: turn k was generated
    conditioned on prompt_k and its captured logprobs are under prompt_k, so training it
    inside a re-rendered sequence would compute trainer logprobs under a different prefix
    and the TIS ratio would mix re-rendering drift with the DeltaNet drift it exists to
    isolate. That is why a masking rule does not rescue the single-sequence path.
    """
    turns = ep.assistant_turns
    if not turns:
        return []

    if episode_is_single_sequence(ep):
        last = turns[-1]
        seq = list(last.prompt_token_ids) + list(last.token_ids)
        mask = [False] * len(seq)
        lps: list[float | None] = [None] * len(seq)
        for t in turns:
            start = len(t.prompt_token_ids)
            for i, lp in enumerate(t.logprobs):
                mask[start + i] = True
                lps[start + i] = lp
        return [Assembled(token_ids=seq, loss_mask=mask, logprobs=lps)]

    out = []
    for t in turns:
        seq = list(t.prompt_token_ids) + list(t.token_ids)
        mask = [False] * len(t.prompt_token_ids) + [True] * len(t.token_ids)
        lps = [None] * len(t.prompt_token_ids) + list(t.logprobs)
        out.append(Assembled(token_ids=seq, loss_mask=mask, logprobs=lps))
    return out


def gate_report(rec_dir: str | Path) -> dict[str, Any]:
    """The prefix-consistency gate. Run over the 256 recorded base-model episodes BEFORE any
    trainer code is written; the answer decides the assembly design."""
    n_ep = n_single = n_turn_pairs = n_prefix_ok = n_mask_ok = 0
    n_trainable = 0
    tokens_single = tokens_per_turn = 0
    terminations: dict[str, int] = {}

    for ep in iter_episodes(rec_dir):
        n_ep += 1
        terminations[ep.termination] = terminations.get(ep.termination, 0) + 1
        if ep.trains:
            n_trainable += 1
        checks = check_episode(ep)
        n_turn_pairs += len(checks)
        n_prefix_ok += sum(c.prefix_ok for c in checks)
        n_mask_ok += sum(c.mask_ok for c in checks)
        if episode_is_single_sequence(ep):
            n_single += 1
        turns = ep.assistant_turns
        if turns:
            tokens_single += len(turns[-1].prompt_token_ids) + len(turns[-1].token_ids)
            tokens_per_turn += sum(len(t.prompt_token_ids) + len(t.token_ids) for t in turns)

    ep_rate = (n_single / n_ep) if n_ep else 0.0
    pair_rate = (n_mask_ok / n_turn_pairs) if n_turn_pairs else 1.0
    saving = (tokens_per_turn / tokens_single) if tokens_single else 0.0

    # An empty directory is "the gate did not run", not "the gate failed". Reporting FAIL
    # here would tell you to go build construction on the strength of zero evidence.
    if n_ep == 0:
        verdict = "NO_DATA"
    elif ep_rate >= GATE_THRESHOLD:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "episodes": n_ep,
        "trainable": n_trainable,
        "terminations": terminations,
        "turn_pairs": n_turn_pairs,
        "prefix_ok_rate": (n_prefix_ok / n_turn_pairs) if n_turn_pairs else 1.0,
        "mask_ok_rate": pair_rate,
        "single_sequence_episodes": n_single,
        "single_sequence_rate": ep_rate,
        "tokens_single_sequence": tokens_single,
        "tokens_per_turn": tokens_per_turn,
        "compute_saving_x": saving,
        "threshold": GATE_THRESHOLD,
        "verdict": verdict,
    }


def failing_spans(rec_dir: str | Path, limit: int = 5) -> list[dict[str, Any]]:
    """The first divergence in each failing episode, as token IDs on both sides.

    This is the payload of a FAIL verdict: if the prefix property breaks, it almost
    certainly breaks at tool-call re-serialization (vLLM parses generated text into a
    structured tool_calls object, tau2 appends the structured form, and the next request's
    template re-serializes it with its own json.dumps key order and whitespace). Knowing
    exactly which tokens moved is what makes driver-side rendering tractable to write,
    instead of reverse-engineering Qwen3.5's template from scratch.
    """
    out: list[dict[str, Any]] = []
    for ep in iter_episodes(rec_dir):
        if episode_is_single_sequence(ep):
            continue
        turns = ep.assistant_turns
        for c in check_episode(ep):
            if c.ok:
                continue
            cur, nxt = turns[c.index], turns[c.index + 1]
            n = len(cur.prompt_token_ids)
            # first index where the carried-over history diverges
            at = next(
                (i for i, (a, b) in enumerate(zip(cur.prompt_token_ids,
                                                  nxt.prompt_token_ids)) if a != b),
                n if not c.prefix_ok else None,
            )
            lo = max(0, (at or n) - 8)
            out.append({
                "episode_id": ep.episode_id,
                "turn_pair": c.index,
                "prefix_ok": c.prefix_ok,
                "mask_ok": c.mask_ok,
                "diverges_at": at,
                "expected": cur.prompt_token_ids[lo : (at or n) + 8],
                "actual": nxt.prompt_token_ids[lo : (at or n) + 8],
                "prior_tool_call": (turns[c.index].tool_call or {}).get("name"),
            })
            break  # first divergence per episode is enough to characterize it
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------------ derived metrics


def auth_before_first_write(ep: Episode) -> bool | None:
    """Authentication check, derived at read time. True if an auth tool was called before the first
    mutating tool call, False if a write happened with no prior auth, None if the episode
    never wrote anything (the question does not apply).

    Kept out of the reward on purpose: the moment it becomes a shaping term, the check is
    measuring its own training signal.
    """
    seen_auth = False
    for t in ep.assistant_turns:
        for call in t.tool_calls or ([t.tool_call] if t.tool_call else []):
            name = call.get("name")
            if name in AUTH_TOOLS:
                seen_auth = True
            elif name is not None and name not in READ_ONLY_TOOLS:
                return seen_auth
    return None


def _write_key(name: str, args: dict[str, Any]) -> tuple[str, Any]:
    # Order tools act on an order; modify_user_address acts on a user.
    return name, args.get("order_id", args.get("user_id"))


def unrequested_writes(ep: Episode, gold_actions: list[dict[str, Any]]) -> int:
    """Successful write calls the task did not ask for: per (tool, order or user), calls
    beyond the gold count. A call whose tool result is an error changed nothing and is not
    counted, so a retry after a rejected call is free. Tool results follow their assistant
    turn in call order, which is how the driver appends them.

    Sept 26: the one shaping term in the training reward (trainer `unrequested_penalty`).
    The DB check already zeroes most extra writes; this separates failures that also broke
    something unasked from plain failures, and gives all-fail groups a gradient against it.
    Never part of eval_reward."""
    gold: dict[tuple[str, Any], int] = {}
    for a in gold_actions:
        if a["name"] not in READ_ONLY_TOOLS:
            k = _write_key(a["name"], a.get("arguments") or {})
            gold[k] = gold.get(k, 0) + 1
    done: dict[tuple[str, Any], int] = {}
    turns = ep.turns
    for i, t in enumerate(turns):
        if not isinstance(t, AssistantTurn):
            continue
        calls = t.tool_calls or ([t.tool_call] if t.tool_call else [])
        results = []
        for r in turns[i + 1:]:
            if isinstance(r, EnvTurn) and r.role == "tool":
                results.append(r)
            else:
                break
        for j, c in enumerate(calls):
            name = c.get("name")
            if name is None or name in READ_ONLY_TOOLS:
                continue
            ok = j < len(results) and not results[j].content.startswith("Error")
            if ok:
                k = _write_key(name, c.get("arguments") or {})
                done[k] = done.get(k, 0) + 1
    return sum(max(0, n - gold.get(k, 0)) for k, n in done.items())


def _successful_writes(ep: Episode) -> list[tuple[str, dict[str, Any]]]:
    out = []
    turns = ep.turns
    for i, t in enumerate(turns):
        if not isinstance(t, AssistantTurn):
            continue
        calls = t.tool_calls or ([t.tool_call] if t.tool_call else [])
        results = []
        for r in turns[i + 1:]:
            if isinstance(r, EnvTurn) and r.role == "tool":
                results.append(r)
            else:
                break
        for j, c in enumerate(calls):
            name = c.get("name")
            if name is None or name in READ_ONLY_TOOLS:
                continue
            if j < len(results) and not results[j].content.startswith("Error"):
                out.append((name, c.get("arguments") or {}))
    return out


def _canon(args: dict[str, Any]) -> str:
    """Arguments as a comparable string; a swap's (item_ids, new_item_ids) become a sorted list
    of pairs, since which item maps to which is what matters, not the order they were listed."""
    a = dict(args)
    if isinstance(a.get("item_ids"), list) and isinstance(a.get("new_item_ids"), list):
        a["pairs"] = sorted(zip(a.pop("item_ids"), a.pop("new_item_ids")))
    elif isinstance(a.get("item_ids"), list):
        a["item_ids"] = sorted(a["item_ids"])
    return json.dumps(a, sort_keys=True)


def gold_write_progress(ep: Episode, gold_actions: list[dict[str, Any]]) -> tuple[int, int, int]:
    """(gold writes matched exactly, gold writes, successful writes that match no gold write).

    Sept 26, the gated partial-credit term of p4-e's training reward (SynthAgent's
    I(no bad action) * fraction of subgoals): a failed episode that did 1 of 2 requested writes
    exactly and nothing else earns partial credit; any write that is not a gold write -- wrong
    order, wrong items, wrong address -- zeroes it. Never part of eval_reward."""
    gold = Counter(_canon_key(a["name"], a.get("arguments") or {})
                   for a in gold_actions if a["name"] not in READ_ONLY_TOOLS)
    done = Counter(_canon_key(n, a) for n, a in _successful_writes(ep))
    matched = sum(min(n, gold[k]) for k, n in done.items())
    return matched, sum(gold.values()), sum(done.values()) - matched


def _canon_key(name: str, args: dict[str, Any]) -> str:
    return name + ":" + _canon(args)


def _write_closeness(gold: dict[str, Any], done: dict[str, Any]) -> float:
    """How close one successful write came to one gold write of the same tool on the same
    target (order or user), in [0, 1]. Swaps score the share of gold (old -> new) pairs done,
    over the larger of the two pair counts, so extra or mis-mapped pairs cost; every other
    argument (payment method, address fields, reason) is a field that matches or does not, and
    a swap with right pairs but the wrong payment method keeps half."""
    g, d = dict(gold), dict(done)
    items = None
    if isinstance(g.get("item_ids"), list) and isinstance(g.get("new_item_ids"), list):
        gp = set(zip(g.pop("item_ids"), g.pop("new_item_ids")))
        dp = set(zip(d.pop("item_ids", []) or [], d.pop("new_item_ids", []) or []))
        items = len(gp & dp) / max(len(gp), len(dp), 1)
    elif isinstance(g.get("item_ids"), list):
        gi, di = set(g.pop("item_ids")), set(d.pop("item_ids", []) or [])
        items = len(gi & di) / max(len(gi), len(di), 1)
    rest = [k for k in g if k not in ("order_id", "user_id")]
    same = sum(g[k] == d.get(k) for k in rest) / len(rest) if rest else 1.0
    if items is None:
        return same                       # address, cancel reason, payment: share of fields
    return items * (1.0 if same == 1.0 else 0.5)


def rejected_writes(ep: Episode) -> int:
    """Write calls the environment rejected (result starting "Error"). They leave the DB alone,
    so a successful episode can carry them; Sept 29, p4-j: 6.7% of successes did, more often on
    tasks solved < 75% of the time (10.9% vs 6.5%), and those ran 16 turns against 13."""
    n = 0
    for i, t in enumerate(ep.turns):
        if not isinstance(t, AssistantTurn):
            continue
        calls = t.tool_calls or ([t.tool_call] if t.tool_call else [])
        results = []
        for r in ep.turns[i + 1:]:
            if isinstance(r, EnvTurn) and r.role == "tool":
                results.append(r)
            else:
                break
        for j, c in enumerate(calls):
            name = c.get("name")
            if name and name not in READ_ONLY_TOOLS and j < len(results) \
                    and results[j].content.startswith("Error"):
                n += 1
    return n


def repeated_calls(ep: Episode) -> int:
    """Sept 29: tool calls (read or write) identical in name and arguments to an earlier call in
    the same episode. RAFT-30B-A3B's GRPO penalizes these, and dropping that penalty cost it
    11.4 pass^4 with pass^1 barely moving (raft.hailab.io): repeats mark the loops and re-checks
    that make a usually-solved task fail one attempt in four."""
    seen: set[str] = set()
    n = 0
    for t in ep.turns:
        if not isinstance(t, AssistantTurn):
            continue
        for c in t.tool_calls or ([t.tool_call] if t.tool_call else []):
            name = c.get("name")
            if not name:
                continue
            key = _canon_key(name, c.get("arguments") or {})
            n += key in seen
            seen.add(key)
    return n


def graded_write_credit(ep: Episode, gold_actions: list[dict[str, Any]]) -> float:
    """Sept 29: argument-level credit in [0, 1] for a FAILED episode, replacing the gate.

    gold_write_progress's gate zeroes the credit on any write that is not exactly gold, and ~70%
    of p4-j's failures had one -- mostly the right order and item swapped to the wrong variant --
    so almost every failure scored like doing nothing. Here each gold write is paired with the
    closest successful write of the same tool on the same order/user (each write used once) and
    scores its closeness; a write with no gold counterpart on its tool and target subtracts a
    full write. Credit = (sum of closeness - unmatched writes) / gold writes, clipped to [0, 1].
    Never part of eval_reward."""
    gold = [a for a in gold_actions if a["name"] not in READ_ONLY_TOOLS]
    if not gold:
        return 0.0
    done = _successful_writes(ep)
    tgt = lambda a: a.get("order_id") or a.get("user_id")  # noqa: E731
    used: set[int] = set()
    total = 0.0
    for g in gold:
        ga = g.get("arguments") or {}
        best, bi = 0.0, None
        for i, (n, a) in enumerate(done):
            if i in used or n != g["name"] or tgt(a) != tgt(ga):
                continue
            c = _write_closeness(ga, a)
            if bi is None or c > best:
                best, bi = c, i
        if bi is not None:
            used.add(bi)
            total += best
    unmatched = len(done) - len(used)
    return max(0.0, min(1.0, (total - unmatched) / len(gold)))


FAILURE_CLASSES = ("runaway", "wrong_target", "missed_write", "wrong_args", "other")


def failure_class(ep: Episode, gold_actions: list[dict[str, Any]]) -> str | None:
    """Why a failed episode failed, by rule (the classes of the Sept 26 transcript read):
    runaway (did not end naturally), wrong_target (a write on an order/user the gold never
    touches), missed_write (a gold write never attempted successfully), wrong_args (every gold
    write attempted on the right target, some with the wrong arguments), other. None on success.
    p4-e's sampler steers task families toward the classes that dominate recent failures."""
    if ep.eval_reward:
        return None
    if ep.termination != "natural":
        return "runaway"
    gold = {_write_key(a["name"], a.get("arguments") or {})
            for a in gold_actions if a["name"] not in READ_ONLY_TOOLS}
    done = [(_write_key(n, a), n, a) for n, a in _successful_writes(ep)]
    if any(k not in gold for k, _, _ in done):
        return "wrong_target"
    if gold - {k for k, _, _ in done}:
        return "missed_write"
    matched, total, wrong = gold_write_progress(ep, gold_actions)
    if wrong or matched < total:
        return "wrong_args"
    return "other"


def task_profile(rec_dir: str | Path) -> dict[str, dict[str, int]]:
    """Per-task success counts over trainable episodes, in the shape carve_val.py --profile
    and the trainer's Beta-posterior seeding both read: {task_id: {successes, trials}}."""
    out: dict[str, dict[str, int]] = {}
    for ep in iter_episodes(rec_dir):
        if not ep.trains:
            continue
        p = out.setdefault(ep.task_id, {"successes": 0, "trials": 0})
        p["trials"] += 1
        p["successes"] += ep.eval_reward
    return dict(sorted(out.items()))


# ------------------------------------------------------------------------------------ CLI


def _fmt_gate(r: dict[str, Any]) -> str:
    lines = [
        f"episodes              {r['episodes']}  ({r['trainable']} trainable)",
        f"assistant turn pairs  {r['turn_pairs']}",
        f"prefix ok             {r['prefix_ok_rate']:.4f}",
        f"mask-alignment ok     {r['mask_ok_rate']:.4f}",
        f"single-sequence eps   {r['single_sequence_episodes']}/{r['episodes']} "
        f"= {r['single_sequence_rate']:.4f}   (threshold {r['threshold']})",
        f"tokens single-seq     {r['tokens_single_sequence']:,}",
        f"tokens per-turn       {r['tokens_per_turn']:,}",
        f"compute saving        {r['compute_saving_x']:.2f}x",
        f"terminations          {r['terminations']}",
        "",
        f"VERDICT: {r['verdict']}",
    ]
    if r["verdict"] == "NO_DATA":
        lines += [
            "",
            "No episodes in that directory, so the gate did not run. This is not a FAIL:",
            "record the 256 base-model episodes first (G=8 x 16 dev tasks x 2 repeats),",
            "then re-run.",
        ]
    elif r["verdict"] == "FAIL":
        lines += [
            "",
            "Below threshold. Do NOT proceed on the per-turn path: it costs ~8x and leaves",
            "~15 optimizer steps. Build construction instead -- driver-side template",
            "rendering plus qwen3_coder parsing, generation via /v1/completions -- which",
            "makes prefix-consistency true by definition.",
            "",
            "Run `records.py spans <dir>` for the token diffs that make that work tractable.",
        ]
    return "\n".join(lines)


def _selftest() -> int:
    """Round-trip plus gate math on synthetic episodes. Needs no GPU, no API key and no
    recorded data, which is the point: the format is developed and tested before the first
    episode exists."""
    import shutil

    tmp = Path(tempfile.mkdtemp(prefix="rec-selftest-"))
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    def mk(eid: str, clean: bool) -> Episode:
        # turn 0: prompt [1,2,3] -> out [10,11]
        # turn 1 prompt is turn 0's prompt + its output + a tool result token.
        # When `clean` is False, byte 1 of the carried-over history differs, which is what
        # tool-call re-serialization looks like at the token level.
        p0 = [1, 2, 3]
        o0 = [10, 11]
        carried = list(p0) + list(o0) if clean else [1, 99, 3] + list(o0)
        p1 = carried + [50]
        o1 = [12]
        return Episode(
            episode_id=eid,
            batch_id=17,
            policy_version="base-b7ea907",
            task_id="tb_0042",
            group_id="b017-t0042",
            trial_index=0,
            seed=1017,
            turns=[
                AssistantTurn(p0, o0, [-0.1, -0.2], "hi", "tool_calls",
                              {"name": "get_user_details", "arguments": {}}),
                EnvTurn("tool", "{}", "get_user_details"),
                AssistantTurn(p1, o1, [-0.3], "bye", "stop",
                              {"name": "return_delivered_order_items", "arguments": {}}),
            ],
            reward=1,
            termination="natural",
            db_hash_before="a" * 8,
            db_hash_after="b" * 8,
            wall_clock_s=41.2,
            recorded_at="2026-09-21T19:00:00Z",
        )

    try:
        clean = mk("ep-clean", clean=True)
        dirty = mk("ep-dirty", clean=False)

        # round trip
        path = write_episode(clean, tmp)
        back = read_episode(path)
        check(back.episode_id == clean.episode_id, "round-trip lost episode_id")
        check(back.assistant_turns[0].token_ids == [10, 11], "round-trip lost token_ids")
        check(isinstance(back.turns[1], EnvTurn), "round-trip lost EnvTurn type")
        check(back.trains is True, "natural should train")

        # atomicity: no temp files survive a successful write
        check(not list(tmp.glob(".tmp-*")), "temp file left behind")

        # checks
        check(episode_is_single_sequence(clean), "clean episode should pass both checks")
        check(not episode_is_single_sequence(dirty), "dirty episode should fail prefix check")
        c = check_episode(dirty)[0]
        check(not c.prefix_ok and not c.mask_ok, "dirty episode should fail both flags")

        # assembly: clean is one sequence, and the mask recovers exactly the generated tokens
        a = assemble(clean)
        check(len(a) == 1, f"clean should assemble to 1 sequence, got {len(a)}")
        got = [t for t, m in zip(a[0].token_ids, a[0].loss_mask) if m]
        check(got == [10, 11, 12], f"single-seq mask recovered {got}, want [10, 11, 12]")
        check(a[0].n_trained_tokens == 3, "wrong trained-token count")

        # assembly: dirty falls back to per-turn, still recovering exactly the generated tokens
        b = assemble(dirty)
        check(len(b) == 2, f"dirty should assemble per-turn into 2, got {len(b)}")
        got_b = [t for s in b for t, m in zip(s.token_ids, s.loss_mask) if m]
        check(got_b == [10, 11, 12], f"per-turn mask recovered {got_b}, want [10, 11, 12]")

        # derived: auth flag. get_user_details is read-only and not an auth tool, so the
        # first write lands with no auth seen.
        check(auth_before_first_write(clean) is False, "auth flag should be False here")

        # derived: unrequested writes. A rejected call is free, a repeat of a gold write and
        # a write to another order are not.
        def call(name: str, **args: Any) -> dict[str, Any]:
            return {"id": name, "name": name, "arguments": args}
        gold_acts = [{"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}]
        import dataclasses
        uw = dataclasses.replace(clean, turns=[
            AssistantTurn([1], [2], [-0.1], "", "tool_calls",
                          tool_calls=[call("cancel_pending_order", order_id="#W1"),
                                      call("modify_user_address", user_id="u1")]),
            EnvTurn("tool", "Error: order is not pending", "cancel_pending_order"),
            EnvTurn("tool", "{...}", "modify_user_address"),
            AssistantTurn([1], [2], [-0.1], "", "tool_calls",
                          tool_calls=[call("cancel_pending_order", order_id="#W1")]),
            EnvTurn("tool", "{...}", "cancel_pending_order"),
        ])
        check(unrequested_writes(uw, gold_acts) == 1, f"unrequested {unrequested_writes(uw, gold_acts)}, want 1")
        check(unrequested_writes(uw, gold_acts + [{"name": "modify_user_address",
                                                   "arguments": {"user_id": "u1"}}]) == 0,
              "a requested address change counted as unrequested")

        # derived: gold-write progress. Swap pairs match whatever order they are listed in; a
        # write matching no gold write is counted as wrong.
        gold2 = [{"name": "modify_pending_order_items",
                  "arguments": {"order_id": "#W1", "item_ids": ["a", "b"], "new_item_ids": ["A", "B"]}},
                 {"name": "modify_pending_order_address", "arguments": {"order_id": "#W1", "zip": "1"}}]
        gp = dataclasses.replace(clean, turns=[
            AssistantTurn([1], [2], [-0.1], "", "tool_calls", tool_calls=[
                call("modify_pending_order_items", order_id="#W1", item_ids=["b", "a"],
                     new_item_ids=["B", "A"])]),
            EnvTurn("tool", "{...}", "modify_pending_order_items"),
        ])
        check(gold_write_progress(gp, gold2) == (1, 2, 0), f"progress {gold_write_progress(gp, gold2)}")
        gq = dataclasses.replace(clean, turns=[
            AssistantTurn([1], [2], [-0.1], "", "tool_calls", tool_calls=[
                call("modify_pending_order_items", order_id="#W1", item_ids=["a", "b"],
                     new_item_ids=["B", "A"])]),
            EnvTurn("tool", "{...}", "modify_pending_order_items"),
        ])
        check(gold_write_progress(gq, gold2) == (0, 2, 1), "mis-paired swap should count as wrong")
        # graded credit: the gold swap done exactly = 1 of 2 writes; the mis-paired swap is on the
        # right order, so it earns its matching pairs (0 of 2) instead of zeroing everything
        check(abs(graded_write_credit(gp, gold2) - 0.5) < 1e-9, f"graded {graded_write_credit(gp, gold2)}")
        check(graded_write_credit(gq, gold2) == 0.0, f"graded mis-paired {graded_write_credit(gq, gold2)}")
        gr = dataclasses.replace(clean, turns=[
            AssistantTurn([1], [2], [-0.1], "", "tool_calls", tool_calls=[
                call("modify_pending_order_items", order_id="#W1", item_ids=["a", "b"],
                     new_item_ids=["A", "X"])]),
            EnvTurn("tool", "{...}", "modify_pending_order_items"),
        ])
        check(abs(graded_write_credit(gr, gold2[:1]) - 0.5) < 1e-9, f"half-right swap {graded_write_credit(gr, gold2[:1])}")
        check(graded_write_credit(uw, gold_acts) == 0.0, "a write to another target must subtract")
        check(rejected_writes(uw) == 1 and rejected_writes(gp) == 0, f"rejected {rejected_writes(uw)}")
        # repeated calls: the same read twice and the same (reordered) swap twice = 2; a
        # different argument is not a repeat
        rp = dataclasses.replace(clean, turns=[
            AssistantTurn([1], [2], [-0.1], "", "tool_calls", tool_calls=[
                call("get_order_details", order_id="#W1"), call("get_order_details", order_id="#W2")]),
            EnvTurn("tool", "{...}", "get_order_details"), EnvTurn("tool", "{...}", "get_order_details"),
            AssistantTurn([1], [2], [-0.1], "", "tool_calls", tool_calls=[
                call("get_order_details", order_id="#W1")]),
            EnvTurn("tool", "{...}", "get_order_details"),
        ] + gp.turns + [
            AssistantTurn([1], [2], [-0.1], "", "tool_calls", tool_calls=[
                call("modify_pending_order_items", order_id="#W1", item_ids=["a", "b"],
                     new_item_ids=["A", "B"])]),
            EnvTurn("tool", "{...}", "modify_pending_order_items"),
        ])
        check(repeated_calls(rp) == 2, f"repeated {repeated_calls(rp)}, want 2")
        check(repeated_calls(gp) == 0, "a single call counted as repeated")
        fail = lambda e: dataclasses.replace(e, reward=0, termination="natural")  # noqa: E731
        check(failure_class(fail(gp), gold2) == "missed_write", f"class {failure_class(fail(gp), gold2)}")
        check(failure_class(fail(gq), gold2[:1]) == "wrong_args", f"class {failure_class(fail(gq), gold2[:1])}")
        check(failure_class(fail(uw), gold_acts) == "wrong_target", f"class {failure_class(fail(uw), gold_acts)}")
        check(failure_class(dataclasses.replace(gp, reward=0, termination="length"), gold2) == "runaway",
              "length termination not runaway")

        # schema guard
        try:
            AssistantTurn([1], [2, 3], [-0.1], "x", "stop")
            failures.append("logprobs/token_ids mismatch not caught")
        except SchemaError:
            pass
        try:
            Episode(**{**asdict(clean), "turns": clean.turns, "termination": "nope"})
            failures.append("bad termination not caught")
        except SchemaError:
            pass

        # gate math over a 1-clean 1-dirty dir -> 0.5, below threshold
        write_episode(dirty, tmp)
        r = gate_report(tmp)
        check(r["episodes"] == 2, f"gate saw {r['episodes']} episodes, want 2")
        check(abs(r["single_sequence_rate"] - 0.5) < 1e-9, "gate rate should be 0.5")
        check(r["verdict"] == "FAIL", "0.5 should FAIL against a 0.95 threshold")
        check(r["compute_saving_x"] > 1.0, "per-turn should cost more tokens than single-seq")

        # an empty dir is NO_DATA, never FAIL: FAIL would send you to build construction
        # on zero evidence
        empty = Path(tempfile.mkdtemp(prefix="rec-empty-"))
        try:
            check(gate_report(empty)["verdict"] == "NO_DATA", "empty dir should be NO_DATA")
        finally:
            shutil.rmtree(empty, ignore_errors=True)

        # failing spans localize the divergence for the construction work
        sp = failing_spans(tmp)
        check(len(sp) == 1, f"expected 1 failing span, got {len(sp)}")
        check(sp[0]["diverges_at"] == 1, f"divergence at {sp[0]['diverges_at']}, want 1")
        check(sp[0]["episode_id"] == "ep-dirty", "wrong episode flagged")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("SELFTEST FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("selftest ok")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().rsplit("CLI:", 1)[-1].strip(), file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "selftest":
        return _selftest()
    if cmd == "gate":
        if len(argv) < 3:
            print("usage: records.py gate <dir>", file=sys.stderr)
            return 2
        r = gate_report(argv[2])
        print(_fmt_gate(r))
        return 0 if r["verdict"] == "PASS" else 1
    if cmd == "spans":
        if len(argv) < 3:
            print("usage: records.py spans <dir>", file=sys.stderr)
            return 2
        spans = failing_spans(argv[2])
        if not spans:
            print("no failing episodes")
            return 0
        for s in spans:
            print(f"{s['episode_id']}  pair {s['turn_pair']}  "
                  f"prefix_ok={s['prefix_ok']} mask_ok={s['mask_ok']}  "
                  f"diverges_at={s['diverges_at']}  after tool={s['prior_tool_call']}")
            print(f"    expected {s['expected']}")
            print(f"    actual   {s['actual']}")
        return 0
    if cmd == "profile":
        if len(argv) < 3:
            print("usage: records.py profile <dir>", file=sys.stderr)
            return 2
        print(json.dumps(task_profile(argv[2]), indent=1))
        return 0
    if cmd == "inspect":
        if len(argv) < 3:
            print("usage: records.py inspect <file>", file=sys.stderr)
            return 2
        ep = read_episode(argv[2])
        print(f"{ep.episode_id}  task={ep.task_id}  policy={ep.policy_version}")
        print(f"  reward={ep.reward}  termination={ep.termination}  trains={ep.trains}")
        print(f"  turns={ep.turn_count}  single_sequence={episode_is_single_sequence(ep)}")
        print(f"  auth_before_first_write={auth_before_first_write(ep)}")
        for i, t in enumerate(ep.turns):
            if isinstance(t, AssistantTurn):
                tc = t.tool_call["name"] if t.tool_call else "-"
                print(f"  [{i}] assistant  prompt={len(t.prompt_token_ids):>6}  "
                      f"gen={len(t.token_ids):>4}  finish={t.finish_reason:<10} tool={tc}")
            else:
                print(f"  [{i}] {t.role:<9}  {t.content[:60]!r}")
        return 0
    print(f"unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
