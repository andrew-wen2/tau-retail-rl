"""Rollout driver for the GRPO loop: the turn state machine that turns a tau2 task
into an episode record, and the batch runner that turns a manifest into a directory of them.

This is the "written myself" core. It owns the turn loop, the guards, token capture, record
assembly and batching. It borrows vLLM (as a server), the tau3 environment, and tau2's user
simulator.

Three things shape the structure:

  * Everything tau2-specific sits behind ONE seam, `TaskRuntime`. `Tau2Retail` binds it to
    tau2 at b7ea907; `ScriptedRuntime` implements the same seam with canned responses, so
    the loop below is fully testable on a laptop with no GPU, no tau2 and no API key.

  * The turn loop mirrors tau2's half-duplex Orchestrator (orchestrator/orchestrator.py at
    b7ea907) wherever a difference could move the reward, because the frozen base arm and
    the final eval both run through that orchestrator:
      - the agent's history opens with tau2's canned "Hi! How can I help you today?"
      - the system prompt is LLMAgent's <instructions>/<policy> template, verbatim
      - every tool call in a turn is executed, and a failing call returns "Error: ..." to
        the agent as a tool result instead of ending the episode (Environment.get_response)
      - max_steps counts every message (agent, customer, tool batch), max 200, and 10 failed
        tool calls end the episode (DEFAULT_MAX_ERRORS)
      - mixed text-plus-tool-call turns are allowed (enforce_communication_protocol is off
        by default), the text is kept in history, and the customer never sees it
      - the customer is tau2's own UserSimulator at temperature 0, seeded per episode
      - messages are shaped like tau2's to_litellm_messages, including tool_call_id
    The one deliberate difference: an empty agent message ends the episode at reward 0
    (the third rollout guard), where tau2 would raise.

  * Requests are NON-STREAMING and carry `return_token_ids` plus `logprobs` with
    `top_logprobs: 0`. Streaming is what breaks return_token_ids under tool calls (vLLM
    #27482, fixed by #29074, streaming only), and top-k logprobs would multiply the record's
    storage by ~20 for values TIS never uses. Because only the sampled-token logprob is
    captured, policy ENTROPY cannot be reconstructed from records and is accumulated online.

Sampling matches the eval arm exactly (baseline/run_baseline.sh THINK_OFF): temperature 0.7,
top_p 0.8, top_k 20, presence_penalty 1.5, thinking off. `top_k` and `chat_template_kwargs`
go at the TOP level of the request body: `extra_body` is an OpenAI-SDK construct that the SDK
flattens, and sent raw it is ignored by vLLM, which would silently turn thinking back on.

API spend is capped: every batch appends its customer cost to a ledger and refuses to start
once the ledger reaches API_BUDGET_USD (default 25).

CLI (run in .venv-tau2):
    python3 driver.py selftest                   full episode + every guard, against fakes
    python3 driver.py batch <manifest.json>      run one manifest, write records, print summary
    python3 driver.py spend                      API spend so far, from the ledger
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

try:
    from records import AssistantTurn, EnvTurn, Episode, write_episode
except ImportError:  # imported as a package
    from .records import AssistantTurn, EnvTurn, Episode, write_episode  # type: ignore

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
TASKS_PATH = TASKS_DIR / "tb500_retail.json"
LEDGER_PATH = Path(os.environ.get("API_LEDGER", HERE / "api_spend.jsonl"))
API_BUDGET_USD = float(os.environ.get("API_BUDGET_USD", "25"))

# tau2 orchestrator.DEFAULT_FIRST_AGENT_MESSAGE at b7ea907. Duplicated rather than imported so
# the driver loop stays importable without tau2; Tau2Retail asserts the two still agree.
GREETING = "Hi! How can I help you today?"

# The training customer (Sept 25: switched from gpt-4.1-nano). DeepSeek-V4-Flash via DeepInfra,
# non-thinking (its default mode), a different family from the Qwen agent. Its cost is computed
# here from token counts: litellm's price table may not know a DeepInfra model, and a customer
# priced at $0 would silently disable the API_BUDGET_USD guard.
CUSTOMER_MODEL = "deepinfra/deepseek-ai/DeepSeek-V4-Flash"
CUSTOMER_PRICES_PER_M = {  # (input, output) USD per 1M tokens, deepinfra.com/pricing Sept 25
    "deepinfra/deepseek-ai/DeepSeek-V4-Flash": (0.09, 0.18),
    # candidate customers for a mixed pool (Sept 28), deepinfra.com model list prices
    "deepinfra/XiaomiMiMo/MiMo-V2.6-Flash": (0.14, 0.28),
    "deepinfra/zai-org/GLM-5.3-Flash": (0.15, 0.50),
    # Sept 29: the OpenAI half of the training customer mix (openai.com/api/pricing); reasoning
    # tokens are billed inside completion_tokens
    "gpt-5-nano": (0.05, 0.40),
    # p4-m: nano role-flipped into the agent on multi-request tasks (0.30 vs DeepSeek 0.68), so
    # a stronger one. Uncached list prices; prompt caching makes the real bill lower.
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4.1-mini": (0.40, 1.60),
}

# The goodbye loop (measured Sept 24). tau2's guidelines say only "generate the '###STOP###'
# token" once the goal is met. gpt-5.2, the eval customer, puts it inside its goodbye in 98.9%
# of episodes and never speaks again; gpt-4.1-nano puts it in a later bare message (99%), keeps
# answering the agent's pleasantries, and in ~20% of episodes never sends it, so solved tasks
# run to the step cap and score 0. Two fixes, both training-customer only:
#   1. the cause: one closing instruction appended to nano's scenario, asking for what gpt-5.2
#      already does. Whether the goal is met stays the customer's judgment.
#   2. a backstop: two consecutive closing-only customer messages end the episode as natural.
# Nothing else about nano changes; its agreeableness is kept on purpose (it is what the customer comparison measures).
#   3. closing-v1 over-corrected: nano attached ###STOP### to "Yes, I confirm. Please proceed"
#      in 14 of 128 episodes, ending them before the agent could act. v2 says when NOT to stop,
#      and the driver ignores a stop on a message that confirms or asks (see premature_stop).
CLOSING_INSTRUCTION = (
    "\n\nEnding the conversation: only after the agent has told you your request is done, or "
    "that it cannot be done, and you have nothing else to ask, say goodbye and include "
    "###STOP### in that same goodbye message. Never include ###STOP### in a message where you "
    "confirm, answer a question, or ask the agent to proceed; wait for the agent to act first. "
    "Do not reply to further pleasantries after your goodbye.")
CUSTOMER_VARIANT = "closing-v2"

# Customer personas (Sept 28). The gpt-5.2 proxy read found most of
# the DeepSeek-trained gain does not carry to the eval customer, and gpt-5.2 opens with twice the
# detail (median 414 characters vs DeepSeek's 199). A training manifest's "personas" ({name:
# weight}) gives each GROUP one style, so a group's G rollouts still differ only by the agent.
# A persona changes how the customer talks, never what it wants. "terse" is the unmodified
# prompt. Val and eval manifests carry no personas.
PERSONAS = {
    "terse": "",
    "forthcoming": (
        "\n\nConversation style: in your first message, say everything you want done and the "
        "details you already know (which order or items, and what should change), in one go. "
        "Then answer the agent's questions directly. This style never changes what you want."),
    "impatient": (
        "\n\nConversation style: you are short on time. Keep your messages brief, ask the agent "
        "to get on with it, and do not volunteer details unless asked. This style never changes "
        "what you want."),
    "uncertain": (
        "\n\nConversation style: you are unsure about the store's rules and about product "
        "details, so before confirming anything ask the agent to explain your options and what "
        "will happen. This style never changes what you want."),
}
MAX_STOP_OVERRIDES = 2

_CLOSING = re.compile(r"\b(thanks?|thank you|bye|goodbye|take care|you too|appreciate|"
                      r"that'?s all|that will be all|that'?s it|nothing else|"
                      r"have a (great|nice|good|wonderful|lovely))\b", re.I)
_NEW_REQUEST = re.compile(r"\?|\b(also|another|actually|but|however|wait|can you|could you|"
                          r"would you|need|want|order|#W\d|confirm|proceed|go ahead)\b", re.I)
_ENDS = re.compile(r"\b(that'?s all|that covers|that will be all|nothing else|good ?bye|bye)\b",
                   re.I)


_PREMATURE = re.compile(r"\?|\b(yes|yeah|go ahead|proceed|confirm|i('d| would) like|i want|"
                        r"can you|could you|please (do|cancel|return|exchange|modify|change|"
                        r"update|process|send|refund|use))\b", re.I)


def premature_stop(text: str) -> bool:
    """A ###STOP### attached to a message that still hands the agent work: a confirmation, a
    request or a question. Ending there would score the agent on a DB it never got to change."""
    t = text.replace("###STOP###", "").strip()
    return "###STOP###" in text and bool(_PREMATURE.search(t)) and not _ENDS.search(t)


def is_closing_only(text: str) -> bool:
    """A customer message that only thanks or says goodbye: short, a closing phrase, and no
    question or new request. Deliberately strict, since a false positive ends a live episode."""
    t = text.replace("###STOP###", "").strip()
    return 0 < len(t) <= 160 and bool(_CLOSING.search(t)) and not _NEW_REQUEST.search(t)


# ----------------------------------------------------------------------------- the guards


@dataclass(frozen=True)
class Guards:
    """Rollout guards. Values and reasoning:

    max_tokens_per_turn kills the baseline-results.md:258 pathology (one unbounded completion
    running 210 tok/s for 35+ minutes) in about ten seconds.

    max_steps and max_errors are tau2's own limits (config.py: DEFAULT_MAX_STEPS 200,
    DEFAULT_MAX_ERRORS 10), counted the way its Orchestrator counts them, so training ends
    episodes on the same condition the eval harness does.

    no_progress_s is NOT a total wall clock. Per-episode latency scales with concurrency, so
    a fixed total budget would fire on healthy episodes at training concurrency and inject
    false reward-0 episodes into the gradient. This bounds a single call instead.
    """

    max_tokens_per_turn: int = 2048
    max_steps: int = 200
    max_errors: int = 10
    no_progress_s: float = 120.0
    customer_retries: int = 2
    # Sept 26 (p4-e, Environment Tuning's "actionable environment augmentation"): append a
    # corrective hint to known tool errors. Training only, and the trainer turns it off for the
    # last steps so the policy ends on the eval environment's plain errors.
    tool_hints: bool = False


SWAP_HINT = (" (Hint: item_ids and new_item_ids are matched by position: the i-th new item replaces"
             " the i-th item and must be a variant of the same product. List only the items that"
             " change; an item cannot be replaced by itself.)")
LOOKUP_HINT = (" (Hint: do not guess ids; look them up from the customer's details and their"
               " orders.)")


def tool_hint(name: str, content: str) -> str:
    """The hint for one tool result, or "" when none applies (only errors get hints)."""
    if not content.startswith("Error"):
        return ""
    if name in ("modify_pending_order_items", "exchange_delivered_order_items"):
        return SWAP_HINT
    if "not found" in content.lower():
        return LOOKUP_HINT
    return ""


# ------------------------------------------------------------------------------ the seam


class TaskRuntime(Protocol):
    """Everything episode-specific the driver needs. One implementation binds tau2; another
    fakes it for tests. The driver loop never imports tau2."""

    task_id: str
    customer_cost: float

    def system_prompt(self) -> str: ...
    def tool_schemas(self) -> list[dict[str, Any]]: ...
    def first_customer_message(self) -> str: ...
    def call_tool(self, call: dict[str, Any]) -> tuple[str, bool]: ...
    def customer_reply(self, assistant_text: str) -> tuple[str, bool]: ...
    def db_hash(self) -> str: ...
    def gold_db_hash(self) -> str: ...


def customer_cost(model: str, msg: Any) -> float:
    """USD for one customer message: from token counts when the model has an explicit price,
    else litellm's own figure. A priced model with no usage reported is charged a
    conservative 8k-in / 1k-out estimate rather than $0, so the budget guard never undercounts."""
    price = CUSTOMER_PRICES_PER_M.get(model)
    if price is None:
        return float(getattr(msg, "cost", None) or 0.0)
    u = getattr(msg, "usage", None) or {}
    get = u.get if isinstance(u, dict) else (lambda k, d=None: getattr(u, k, d))
    pin, pout = get("prompt_tokens"), get("completion_tokens")
    if pin is None or pout is None:
        pin, pout = 8000, 1000
    return (pin * price[0] + pout * price[1]) / 1e6


def _apply_initial_state(env: Any, init: Any) -> None:
    if init is None:
        return
    if getattr(init, "message_history", None):
        raise ValueError("tasks that start mid-conversation are not supported by the driver")
    if init.initialization_data is None and not init.initialization_actions:
        return
    env.set_state(initialization_data=init.initialization_data,
                  initialization_actions=init.initialization_actions, message_history=[])


class Tau2Retail:
    """tau2 retail binding. Runs in .venv-tau2, pinned at b7ea907. Each instance owns a fresh
    environment, so G concurrent rollouts of one task never share a store DB."""

    def __init__(self, task: dict[str, Any], customer_model: str = CUSTOMER_MODEL,
                 seed: int | None = None, customer_args: dict[str, Any] | None = None,
                 closing: bool = True, persona: str | None = None) -> None:
        from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT  # noqa: PLC0415
        from tau2.data_model.tasks import Task  # noqa: PLC0415
        from tau2.domains.retail.environment import get_environment  # noqa: PLC0415
        from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE  # noqa: PLC0415
        from tau2.user.user_simulator import UserSimulator  # noqa: PLC0415

        if DEFAULT_FIRST_AGENT_MESSAGE.content != GREETING:
            raise RuntimeError("tau2's opening agent message changed; update GREETING")

        self.task = task
        self.task_id = task["id"]
        self.customer_model = customer_model
        self.customer_variant = CUSTOMER_VARIANT
        self.customer_cost = 0.0
        self.env = get_environment()
        # Tasks may carry their own database delta (tau2-gen: new customers, orders, products,
        # deep-merged into the default store). The converted tasks carry none. Applied here AND to
        # the gold-replay env, or a task would run and be scored on the wrong database.
        self._init = Task.model_validate(task).initial_state
        _apply_initial_state(self.env, self._init)
        self._system = SYSTEM_PROMPT.format(domain_policy=self.env.get_policy(),
                                            agent_instruction=AGENT_INSTRUCTION)
        self._tools = [t.openai_schema for t in self.env.get_tools()]
        self._gold_hash: str | None = None

        # build_user() at b7ea907: instructions are str(task.user_scenario), retail has no
        # user tools, llm_args default to temperature 0.0, and the Orchestrator seeds it.
        # The one addition is CLOSING_INSTRUCTION (see its comment).
        t = Task.model_validate(task)
        # customer_args/closing=False reproduce the eval customer exactly (gpt-5.2 at low
        # reasoning, tau2's own instructions): the Sept 28 customer-transfer proxy.
        self._user = UserSimulator(llm=customer_model,
                                   instructions=str(t.user_scenario)
                                   + PERSONAS[persona or "terse"]
                                   + (CLOSING_INSTRUCTION if closing else ""),
                                   tools=None, llm_args=customer_args or {"temperature": 0.0})
        if seed is not None:
            self._user.set_seed(seed)
        self._user_state = self._user.get_init_state()

    def system_prompt(self) -> str:
        return self._system

    def tool_schemas(self) -> list[dict[str, Any]]:
        return self._tools

    def _customer(self, agent_text: str) -> tuple[str, bool]:
        from tau2.data_model.message import AssistantMessage  # noqa: PLC0415
        from tau2.user.user_simulator import UserSimulator  # noqa: PLC0415

        msg, self._user_state = self._user.generate_next_message(
            AssistantMessage(role="assistant", content=agent_text), self._user_state)
        self.customer_cost += customer_cost(self.customer_model, msg)
        return msg.content or "", UserSimulator.is_stop(msg)

    def first_customer_message(self) -> str:
        # The Orchestrator's first step sends the canned greeting to the user simulator.
        text, _ = self._customer(GREETING)
        return text

    def customer_reply(self, assistant_text: str) -> tuple[str, bool]:
        return self._customer(assistant_text)

    def call_tool(self, call: dict[str, Any]) -> tuple[str, bool]:
        from tau2.data_model.message import ToolCall  # noqa: PLC0415

        msg = self.env.get_response(ToolCall(id=call.get("id") or "", name=call["name"],
                                             arguments=call["arguments"],
                                             requestor="assistant"))
        return msg.content, bool(msg.error)

    def db_hash(self) -> str:
        return self.env.get_db_hash()

    def communicate_info(self) -> list[str]:
        ec = self.task.get("evaluation_criteria") or {}
        return list(ec.get("communicate_info") or []) if "COMMUNICATE" in (
            ec.get("reward_basis") or ["DB", "COMMUNICATE"]) else []

    def gold_db_hash(self) -> str:
        """Replay the gold actions in a SEPARATE fresh env and hash the result, as tau2's DB
        evaluator does. Every kept task's gold replay was validated error-free at conversion
        (convert_tb.py), so an exception here means the env or the task file moved and the
        reward would be meaningless."""
        if self._gold_hash is not None:
            return self._gold_hash
        from tau2.domains.retail.environment import get_environment  # noqa: PLC0415

        gold = get_environment()
        _apply_initial_state(gold, self._init)
        for action in self.task["evaluation_criteria"]["actions"]:
            gold.make_tool_call(tool_name=action["name"], requestor=action["requestor"],
                                **action["arguments"])
        self._gold_hash = gold.get_db_hash()
        return self._gold_hash


# ------------------------------------------------------------------------- vLLM transport


class VLLMError(RuntimeError):
    pass


@dataclass
class VLLMClient:
    """Non-streaming chat completions with token IDs. stdlib only."""

    base_url: str = "http://localhost:8000/v1"
    model: str = "qwen3.5-2B"  # or the adapter name ("policy") once training starts

    # Eval-arm sampling, verbatim from baseline/run_baseline.sh THINK_OFF.
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 1.5

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                 max_tokens: int, timeout_s: float, seed: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",       # tau2's generate() sets this whenever tools exist
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,             # streaming is what breaks return_token_ids on tool calls
            "return_token_ids": True,
            "logprobs": True,
            "top_logprobs": 0,           # sampled token only; TIS needs nothing more
        }
        if seed is not None:
            body["seed"] = seed          # tau2 seeds the agent per trial; so does this
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise VLLMError(f"HTTP {e.code}: {e.read()[:400]!r}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise TimeoutError(str(e)) from e


@dataclass
class APIClient(VLLMClient):
    """An API model as the AGENT (Sept 28: a stronger-model check on the tasks the 2B fails,
    and the source of fine-tuning transcripts). OpenAI-compatible, keyed from the environment.

    It returns no token IDs, so its episodes never train by RL: the response is given empty
    token_ids/logprobs, which parse_completion accepts (0 == 0) and the records mark as such.
    Spend goes straight to the ledger per call, and the batch budget guard is enforced here
    too, since run_batch's own guard only sees customer cost."""

    base_url: str = "https://api.deepinfra.com/v1/openai"
    model: str = "deepseek-ai/DeepSeek-V4.1-Flash"
    key_env: str = "DEEPINFRA_API_KEY"
    temperature: float = 0.7
    top_p: float = 0.95
    prices_per_m: tuple[float, float] = (0.20, 0.60)  # deepinfra.com model list, Sept 28

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._base = spent_usd()
        self.usd = 0.0

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                 max_tokens: int, timeout_s: float, seed: int | None = None) -> dict[str, Any]:
        if self._base + self.usd >= API_BUDGET_USD:
            raise VLLMError(f"agent API budget: ledger >= ${API_BUDGET_USD:.2f}")
        body: dict[str, Any] = {"model": self.model, "messages": messages, "tools": tools,
                                "tool_choice": "auto", "max_tokens": max_tokens,
                                "temperature": self.temperature, "top_p": self.top_p,
                                "stream": False}
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ[self.key_env]}"})
        last: Exception | None = None
        for attempt in range(4):  # API 429/5xx are infrastructure, not policy
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    out = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                last = VLLMError(f"HTTP {e.code}: {e.read()[:400]!r}")
                if e.code not in (429, 500, 502, 503, 504):
                    raise last from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = TimeoutError(str(e))
            time.sleep(2.0 * (attempt + 1))
        else:
            raise last  # type: ignore[misc]
        u = out.get("usage") or {}
        usd = u.get("estimated_cost")
        if usd is None:
            usd = (u.get("prompt_tokens", 0) * self.prices_per_m[0]
                   + u.get("completion_tokens", 0) * self.prices_per_m[1]) / 1e6
        record_spend(usd, f"agent {self.model}")
        with self._lock:
            self.usd += usd
        out["prompt_token_ids"] = []
        out["choices"][0]["token_ids"] = []
        out["choices"][0]["logprobs"] = None
        return out


def communicated(agent_texts: list[str], infos: list[str]) -> bool:
    """tau2's rule: each info string, lowercased, is a substring of some agent message lowercased
    with commas removed. Vacuously true with nothing to communicate."""
    msgs = [(x or "").lower().replace(",", "") for x in agent_texts]
    return all(any(i.lower() in m for m in msgs) for i in infos)


@dataclass
class Parsed:
    turn: AssistantTurn
    content: str | None                 # exactly what the server returned, None included
    calls: list[dict[str, Any]]         # [{"id", "name", "arguments"}]
    parse_error: bool = False           # tool-call arguments were not valid JSON


def parse_completion(resp: dict[str, Any]) -> Parsed:
    """Pull one AssistantTurn out of a chat-completions response.

    Fails loudly when token IDs are absent. Silently falling back to re-tokenizing the text
    is the exact failure requirement 2 exists to prevent: re-tokenized text is not what the
    model generated, and the loss would land on tokens the policy never emitted.
    """
    choice = resp["choices"][0]
    msg = choice["message"]

    prompt_ids = resp.get("prompt_token_ids")
    token_ids = choice.get("token_ids")
    if prompt_ids is None or token_ids is None:
        raise VLLMError(
            "response carried no token IDs. Check that the server accepts "
            "return_token_ids and that the request is non-streaming; do NOT re-tokenize "
            "the text as a fallback.")

    lps = [e["logprob"] for e in (choice.get("logprobs") or {}).get("content") or []]
    if len(lps) != len(token_ids):
        # A mismatch here would silently misalign every TIS ratio in the episode.
        raise VLLMError(
            f"logprobs/token_ids misaligned: {len(lps)} vs {len(token_ids)}. "
            "Check --logprobs-mode and that top_logprobs=0 returns one entry per token.")

    calls: list[dict[str, Any]] = []
    parse_error = False
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc["function"]
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                parse_error = True
                args = {"_unparsed": args}
        calls.append({"id": tc.get("id") or f"call_{i}", "name": fn["name"],
                      "arguments": args or {}})

    turn = AssistantTurn(
        prompt_token_ids=list(prompt_ids),
        token_ids=list(token_ids),
        logprobs=lps,
        text=msg.get("content") or "",
        finish_reason=choice.get("finish_reason") or "stop",
        tool_call=({"name": calls[0]["name"], "arguments": calls[0]["arguments"]}
                   if calls else None),
        tool_calls=calls or None,
    )
    return Parsed(turn=turn, content=msg.get("content"), calls=calls, parse_error=parse_error)


def _assistant_message(content: str | None, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """tau2 to_litellm_messages() shape for an AssistantMessage, plus one marker.

    `reasoning_content: ""` tells baseline/qwen35_train.jinja (served by serve-train.sh) to
    keep this turn's empty <think></think> block when it re-renders history. The stock
    template strips it from every turn before the latest customer message, which made each
    prompt stop being a token prefix of the next at every customer turn (smoke run, Sept 23:
    7.7% of turn pairs consistent). With the marker, history is rendered exactly as it was
    generated. tau2's canned greeting carries no marker, so it renders as it does in eval.
    """
    return {
        "role": "assistant",
        "reasoning_content": "",
        "content": content,
        "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for c in calls
        ] or None,
    }


# -------------------------------------------------------------------------- the turn loop


@dataclass
class EpisodeIds:
    episode_id: str
    batch_id: int
    policy_version: str
    group_id: str
    trial_index: int
    seed: int


@dataclass
class EpisodeStats:
    """Online metrics that records cannot reconstruct."""

    sum_neg_logprob: float = 0.0
    n_tokens: int = 0
    empty_messages: int = 0
    tool_errors: int = 0
    customer_retries: int = 0

    @property
    def mean_entropy_proxy(self) -> float:
        """Mean -logp of sampled tokens. Not the true entropy, which top_logprobs=0 cannot
        give; it is the standard cheap proxy and it is monotone with entropy collapse, which
        is the thing being watched for."""
        return self.sum_neg_logprob / self.n_tokens if self.n_tokens else 0.0


def run_episode(runtime: TaskRuntime, client: VLLMClient, ids: EpisodeIds,
                guards: Guards = Guards()) -> tuple[Episode, EpisodeStats]:
    """Run one rollout to termination and return an Episode ready for write_episode().

    Never raises for a policy failure: every way an episode can end badly is a termination
    class in the record, because a rollout that disappears silently changes the group size
    and therefore the advantage denominator.
    """
    stats = EpisodeStats()
    turns: list[AssistantTurn | EnvTurn] = []
    db_before = runtime.db_hash()
    started = time.monotonic()

    try:
        opening = runtime.first_customer_message()
    except Exception as e:  # noqa: BLE001
        return _abort(ids, runtime.task_id, db_before, "customer_api_error",
                      time.monotonic() - started, f"opening: {e!r}"), stats

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": runtime.system_prompt()},
        {"role": "assistant", "content": GREETING, "tool_calls": None},
        {"role": "user", "content": opening},
    ]
    turns.append(EnvTurn(role="user", content=opening))
    tools = runtime.tool_schemas()

    # tau2 step accounting: the opening customer message was step 1. Every agent message,
    # tool batch and customer message is one step; limits are checked after each step
    # except while a tool batch is pending, exactly as Orchestrator._check_termination does.
    steps, errors = 1, 0
    termination = "natural"
    prev_closing = closed_by_backstop = False
    stop_overrides = 0

    def over_limit() -> str | None:
        if steps >= guards.max_steps:
            return "max_turns"
        if errors >= guards.max_errors:
            return "too_many_errors"
        return None

    while True:
        # --- policy turn ---
        try:
            resp = client.complete(messages, tools, guards.max_tokens_per_turn,
                                   guards.no_progress_s, seed=ids.seed)
            parsed = parse_completion(resp)
        except TimeoutError:
            # Nothing came back for the in-flight turn (non-streaming), so there is nothing
            # to discard; completed turns still train at reward 0.
            termination = "no_progress_timeout"
            break
        except VLLMError as e:
            turns.append(EnvTurn(role="error", content=str(e)[:500]))
            termination = "env_exception"
            break

        turn = parsed.turn
        turns.append(turn)
        steps += 1
        stats.n_tokens += len(turn.token_ids)
        stats.sum_neg_logprob += -sum(turn.logprobs)

        if turn.finish_reason == "length":
            # Terminal, trains at reward 0 (see records.TERMINATION_CLASSES for why it is
            # not excluded).
            termination = "length"
            break
        if parsed.parse_error:
            termination = "tool_parse_error"
            break
        if not turn.text.strip() and not parsed.calls:
            # Terminal, reward 0, counted, never retried (the third rollout guard).
            stats.empty_messages += 1
            termination = "empty_message"
            break

        messages.append(_assistant_message(parsed.content, parsed.calls))

        # --- tool batch: to_role == ENV, so no limit check before it runs ---
        if parsed.calls:
            for call in parsed.calls:
                content, err = runtime.call_tool(call)
                if guards.tool_hints:
                    content += tool_hint(call["name"], content)
                errors += err
                stats.tool_errors += err
                turns.append(EnvTurn(role="tool", content=content, tool_name=call["name"]))
                messages.append({"role": "tool", "content": content,
                                 "tool_call_id": call["id"]})
            steps += 1
            if (lim := over_limit()) is not None:
                termination = lim
                break
            continue

        if (lim := over_limit()) is not None:
            termination = lim
            break

        # --- customer turn ---
        reply, done = None, False
        for attempt in range(guards.customer_retries + 1):
            try:
                reply, done = runtime.customer_reply(turn.text)
                break
            except Exception:  # noqa: BLE001
                stats.customer_retries += 1
                if attempt < guards.customer_retries:
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
        if reply is None:
            # Infrastructure, not policy: re-rolled in-batch so the group ships at full G.
            termination = "customer_api_error"
            break

        if done and stop_overrides < MAX_STOP_OVERRIDES and premature_stop(reply):
            # The customer confirmed or asked for something AND stopped: drop the stop so the
            # agent gets the turn it was asked to take. Bounded, so it cannot become a loop.
            reply, done = reply.replace("###STOP###", "").strip(), False
            stop_overrides += 1
        turns.append(EnvTurn(role="user", content=reply))
        messages.append({"role": "user", "content": reply})
        steps += 1
        if done:
            termination = "natural"
            break
        # Goodbye-loop backstop: the customer has closed twice in a row with the agent's
        # reply in between, so the conversation is over whether or not nano says ###STOP###.
        closing = is_closing_only(reply)
        if closing and prev_closing:
            termination = "natural"
            closed_by_backstop = True
            break
        prev_closing = closing
        if (lim := over_limit()) is not None:
            termination = lim
            break

    db_after = runtime.db_hash()
    reward = 0
    # tau2's evaluator (evaluator.py:119 at b7ea907) scores 0 for any run the customer or
    # agent did not end: max_steps, too_many_errors and timeouts all get 0 whatever the DB
    # says. Mirrored here. Before Sept 23 the driver scored those on the DB hash, which gave
    # nano's goodbye loops (~26% of base episodes) reward the eval would never give them.
    db_ok = comm_ok = None
    if termination == "natural":
        try:
            db_ok = db_after == runtime.gold_db_hash()
        except Exception:  # noqa: BLE001
            # Gold replay failing means the env or task file moved since conversion, so the
            # reward is meaningless rather than zero. Do not train on a fabricated 0.
            termination = "env_exception"
        else:
            # tau2's CommunicateEvaluator (b7ea907): every communicate_info string must appear,
            # case-insensitively, in some agent message with commas removed; the task reward is
            # DB x COMMUNICATE. Until Sept 28 the driver scored the DB alone, so 349 of 1,742
            # training tasks and 104 of valbig's 166 were graded more leniently than tau3.
            infos = getattr(runtime, "communicate_info", lambda: [])()
            comm_ok = communicated([t.text for t in turns if isinstance(t, AssistantTurn)], infos)
            reward = int(db_ok and comm_ok)

    ep = Episode(
        episode_id=ids.episode_id, batch_id=ids.batch_id, policy_version=ids.policy_version,
        task_id=runtime.task_id, group_id=ids.group_id, trial_index=ids.trial_index,
        seed=ids.seed, turns=turns, reward=reward, termination=termination,
        db_hash_before=db_before, db_hash_after=db_after,
        wall_clock_s=round(time.monotonic() - started, 3),
        recorded_at=datetime.now(timezone.utc).isoformat(),
        extra={
            "steps": steps,
            "db_ok": db_ok,
            "communicate_ok": comm_ok,
            "persona": getattr(runtime, "persona", None),
            "customer": getattr(runtime, "customer_name", None),
            "empty_messages": stats.empty_messages,
            "tool_errors": stats.tool_errors,
            "customer_retries": stats.customer_retries,
            "customer_cost": round(getattr(runtime, "customer_cost", 0.0), 6),
            "mean_neg_logprob": round(stats.mean_entropy_proxy, 6),
            "n_gen_tokens": stats.n_tokens,
            "customer_variant": getattr(runtime, "customer_variant", None),
            "closed_by_backstop": closed_by_backstop,
            "stop_overrides": stop_overrides,
        },
    )
    return ep, stats


def _abort(ids: EpisodeIds, task_id: str, before: str, termination: str,
           elapsed: float, note: str) -> Episode:
    return Episode(
        episode_id=ids.episode_id, batch_id=ids.batch_id, policy_version=ids.policy_version,
        task_id=task_id, group_id=ids.group_id, trial_index=ids.trial_index, seed=ids.seed,
        turns=[], reward=0, termination=termination, db_hash_before=before,
        db_hash_after=before, wall_clock_s=round(elapsed, 3),
        recorded_at=datetime.now(timezone.utc).isoformat(), extra={"abort": note})


# ---------------------------------------------------------------------------- API ledger


_ledger_lock = threading.Lock()


def spent_usd() -> float:
    if not LEDGER_PATH.exists():
        return 0.0
    total = 0.0
    for line in LEDGER_PATH.read_text().splitlines():
        # A Spot preemption can leave the unflushed tail of an append as NUL bytes. The ledger is
        # a spend guard, not a record, so a lost line undercounts by one episode's cost at most.
        line = line.strip().strip("\x00")
        if not line:
            continue
        try:
            total += json.loads(line).get("usd", 0.0)
        except json.JSONDecodeError:
            print(f"warning: skipping corrupt ledger line: {line[:80]!r}", file=sys.stderr)
    return total


def record_spend(usd: float, what: str) -> None:
    with _ledger_lock, LEDGER_PATH.open("a") as fh:
        fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                             "usd": round(usd, 6), "what": what}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class BudgetExceeded(RuntimeError):
    pass


# ---------------------------------------------------------------------------- batch runner


def load_tasks(path: Path | None = None) -> dict[str, dict[str, Any]]:
    return {t["id"]: t for t in json.loads((path or TASKS_PATH).read_text())}


def _load_dotenv() -> None:
    """tau2 does not load its own .env; run_baseline.sh sources it. Do the same here, plus
    `.env.deepinfra` at the repo root (git-ignored, synced by vm.sh) for the customer's key.
    Only fills variables that are not already set."""
    for p in (HERE / "baseline" / "tau2-bench" / ".env", HERE / "tau2-bench" / ".env",
              HERE / ".env.deepinfra"):
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run_batch(manifest: dict[str, Any], runtime_factory=None,
              client: VLLMClient | None = None) -> dict[str, Any]:
    """Run one manifest to completion: G rollouts per task, in-batch re-rolls for
    infrastructure failures, and whole-group drops when a group cannot be refilled.

    Manifest keys: batch_id, policy_version, model, tasks, G, seed, out_dir; optional
    concurrency, customer_model, customer_args, closing_instruction, agent_api, personas,
    customers, base_url, max_rerolls, guards, tasks_file.

    `customers` (Sept 29), {name: {"model", "args", "weight"}}: a customer mix, one customer
    per GROUP drawn by weight, so a group's G rollouts still differ only by the agent. It
    overrides customer_model/customer_args for every group; records carry extra.customer.
    """
    batch_id = int(manifest["batch_id"])
    G = int(manifest["G"])
    out_dir = Path(manifest["out_dir"])
    conc = int(manifest.get("concurrency", 32))
    max_rerolls = int(manifest.get("max_rerolls", 2))
    customer_model = manifest.get("customer_model", CUSTOMER_MODEL)
    guards = Guards(**manifest.get("guards", {}))
    if client is None and manifest.get("agent_api"):
        _load_dotenv()  # the agent's key, before the first request
        client = APIClient(model=manifest["model"])
    client = client or VLLMClient(base_url=manifest.get("base_url", "http://localhost:8000/v1"),
                                  model=manifest["model"])

    before = spent_usd()
    if before >= API_BUDGET_USD:
        raise BudgetExceeded(f"ledger at ${before:.2f} >= budget ${API_BUDGET_USD:.2f}")

    if runtime_factory is None:
        _load_dotenv()
        # tasks_file: e.g. tb500_descripted.json (same ids, tau3-style instructions, same gold)
        tasks = load_tasks(TASKS_DIR / manifest["tasks_file"] if manifest.get("tasks_file") else None)
        def runtime_factory(task_id: str, seed: int, persona: str | None = None,  # noqa: E306
                            customer: str | None = None):
            spec = customers[customer] if customer else {}
            rt = Tau2Retail(tasks[task_id], customer_model=spec.get("model", customer_model),
                            seed=seed,
                            customer_args=spec.get("args", manifest.get("customer_args")),
                            closing=manifest.get("closing_instruction", True),
                            persona=persona)
            rt.persona = persona
            rt.customer_name = customer
            return rt

    customers = manifest.get("customers") or {}
    for name, spec in customers.items():
        if not spec.get("model") or spec.get("weight", 0) <= 0:
            raise ValueError(f"customer {name!r} needs a model and a positive weight: {spec}")
    rng = random.Random(int(manifest["seed"]))
    # A separate stream, so a manifest without personas draws exactly the seeds it always did.
    prng = random.Random(int(manifest["seed"]) ^ 0x5EED)
    personas = manifest.get("personas") or {}
    for name in personas:
        if name not in PERSONAS:
            raise ValueError(f"unknown persona {name!r}; known: {sorted(PERSONAS)}")
    jobs = []
    crng = random.Random(int(manifest["seed"]) ^ 0xC057)
    for gi, task_id in enumerate(manifest["tasks"]):
        group_id = f"b{batch_id:04d}-g{gi:03d}-{task_id}"
        persona = (prng.choices(list(personas), weights=list(personas.values()))[0]
                   if personas else None)
        customer = (crng.choices(list(customers), weights=[c["weight"] for c in customers.values()])[0]
                    if customers else None)
        for k in range(G):
            jobs.append((task_id, group_id, k, rng.randrange(2**31), persona, customer))

    spend_lock = threading.Lock()
    spend = {"usd": 0.0}
    abort = threading.Event()

    def one(job):
        task_id, group_id, k, seed, persona, customer = job
        kw = {k2: v for k2, v in (("persona", persona), ("customer", customer)) if v}
        for attempt in range(max_rerolls + 1):
            if abort.is_set():
                return None
            ids = EpisodeIds(episode_id=f"{group_id}-k{k}", batch_id=batch_id,
                             policy_version=manifest["policy_version"], group_id=group_id,
                             trial_index=k, seed=seed + attempt)
            try:
                rt = runtime_factory(task_id, ids.seed, **kw)
                ep, st = run_episode(rt, client, ids, guards)
            except Exception as e:  # noqa: BLE001 - env construction etc.
                ep, st = _abort(ids, task_id, "", "env_exception", 0.0, repr(e)[:300]), None
            cost = ep.extra.get("customer_cost", 0.0)
            # Per episode, not per batch: a preempted or killed batch must still be counted.
            record_spend(cost, f"batch {batch_id} {ids.episode_id}")
            with spend_lock:
                spend["usd"] += cost
                if before + spend["usd"] >= API_BUDGET_USD:
                    abort.set()
            if not ep.needs_reroll:
                return ep, st
        return ep, st  # still infrastructure-failed after re-rolls; caller drops the group

    # Each group is written the moment its G-th rollout lands, not at the end of the batch,
    # so a preemption loses only the groups still in flight (Sept 23: one landed 75 episodes
    # into a 1,808-episode profiling chunk that wrote nothing).
    results: dict[str, list] = {}
    kept_groups = dropped_groups = 0
    n_written = 0
    term: dict[str, int] = {}
    ent_num = ent_den = 0.0
    rewards: list[int] = []

    def settle(items: list) -> None:
        nonlocal kept_groups, dropped_groups, n_written, ent_num, ent_den
        eps = [e for e, _ in items]
        for e in eps:
            term[e.termination] = term.get(e.termination, 0) + 1
        if len(eps) < G or any(e.needs_reroll for e in eps):
            dropped_groups += 1
            return
        kept_groups += 1
        for e, st in items:
            write_episode(e, out_dir)
            n_written += 1
            if e.trains:
                rewards.append(e.reward)
            if st is not None:
                ent_num += st.sum_neg_logprob
                ent_den += st.n_tokens

    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            if r is None:
                continue
            items = results.setdefault(r[0].group_id, [])
            items.append(r)
            if len(items) == G:
                settle(items)
                del results[r[0].group_id]
    for items in results.values():  # groups cut short by a budget abort
        settle(items)

    summary = {
        "batch_id": batch_id,
        "policy_version": manifest["policy_version"],
        "episodes_written": n_written,
        "groups_kept": kept_groups,
        "groups_dropped": dropped_groups,
        "aborted_on_budget": abort.is_set(),
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "entropy_proxy": (ent_num / ent_den) if ent_den else None,
        "terminations": term,
        "customer_usd": round(spend["usd"], 4),
        "ledger_usd": round(before + spend["usd"], 4),
    }
    (out_dir / f"_batch{batch_id:04d}.summary.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / f"_batch{batch_id:04d}.summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# --------------------------------------------------------------------- offline test double


class ScriptedRuntime:
    """A TaskRuntime backed by canned responses, so the turn loop and every guard can be
    exercised on a laptop with no tau2, no GPU and no API key."""

    def __init__(self, task_id: str = "tb_0042",
                 customer_script: list[tuple[str, bool]] | None = None,
                 tool_results: dict[str, Any] | None = None, gold: str = "GOLD",
                 fail_customer: bool = False, cost: float = 0.0) -> None:
        self.task_id = task_id
        self.customer_cost = 0.0
        self._cost = cost
        self._customer = list(customer_script or [("anything else?", False), ("thanks", True)])
        self._tools = tool_results or {}
        self._gold = gold
        self._hash = "START"
        self._fail_customer = fail_customer

    def system_prompt(self) -> str:
        return "retail policy"

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "get_order_details"}}]

    def first_customer_message(self) -> str:
        self.customer_cost += self._cost
        return "hi, I want to return an order"

    def call_tool(self, call: dict[str, Any]) -> tuple[str, bool]:
        # Mirrors Environment.get_response: failures come back as content, never raise.
        name = call["name"]
        if name not in ("get_order_details", "cancel_pending_order", "think"):
            return f"Error: Tool '{name}' not found.", True
        if name == "cancel_pending_order":
            self._hash = self._gold
        return json.dumps(self._tools.get(name, {"ok": True})), False

    def customer_reply(self, assistant_text: str) -> tuple[str, bool]:
        if self._fail_customer:
            raise ConnectionError("nano unreachable")
        self.customer_cost += self._cost
        return self._customer.pop(0) if self._customer else ("bye", True)

    def db_hash(self) -> str:
        return self._hash

    def gold_db_hash(self) -> str:
        return self._gold


class ScriptedClient(VLLMClient):
    """Replays canned completions and builds token IDs so that the prefix-consistency
    property holds, which is what the real server is being tested for."""

    def __init__(self, script: list[dict[str, Any]], raise_on: Exception | None = None) -> None:
        super().__init__()
        self.script = list(script)
        self.raise_on = raise_on
        self.calls = 0
        self.last_body: dict[str, Any] | None = None
        self._prompt: list[int] = [1, 2, 3]
        self._lock = threading.Lock()

    def complete(self, messages, tools, max_tokens, timeout_s, seed=None):  # type: ignore[override]
        if self.raise_on is not None:
            raise self.raise_on
        with self._lock:
            self.calls += 1
            self.last_body = {"messages": messages, "seed": seed}
            item = self.script.pop(0) if self.script else {"text": "bye", "n": 1}
            n = item.get("n", 2)
            out = [100 + self.calls * 10 + i for i in range(n)]
            prompt = list(self._prompt)
            self._prompt = prompt + out + [50]  # append-only
        return {
            "prompt_token_ids": prompt,
            "choices": [{
                "message": {"content": item.get("text", ""),
                            "tool_calls": item.get("tool_calls")},
                "token_ids": out,
                "logprobs": {"content": [{"logprob": -0.1 * (i + 1)} for i in range(n)]},
                "finish_reason": item.get("finish_reason", "stop"),
            }],
        }


# ------------------------------------------------------------------------------- selftest


def _selftest() -> int:
    import shutil
    import tempfile

    global LEDGER_PATH
    try:
        from records import check_episode, episode_is_single_sequence, read_episode
    except ImportError:
        from .records import check_episode, episode_is_single_sequence, read_episode  # type: ignore

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    def ids(eid: str = "ep-1") -> EpisodeIds:
        return EpisodeIds(eid, 17, "base-b7ea907", "g-1", 0, 1017)

    def tc(name: str = "cancel_pending_order", args: str = '{"order_id":"#W1"}', i: str = "c1"):
        return {"id": i, "type": "function", "function": {"name": name, "arguments": args}}

    # 1. happy path: tool call, then a closing message, reward from the gold hash
    cl = ScriptedClient([{"text": "", "tool_calls": [tc()]}, {"text": "done"}])
    ep, st = run_episode(ScriptedRuntime(customer_script=[("ok thanks", True)]), cl, ids())
    check(ep.termination == "natural", f"happy path terminated {ep.termination}")
    check(ep.reward == 1, f"happy path reward {ep.reward}, want 1")
    check(ep.turn_count == 2, f"happy path {ep.turn_count} assistant turns, want 2")
    check(ep.trains, "happy path should train")
    check(st.n_tokens > 0 and st.mean_entropy_proxy > 0, "entropy proxy not accumulated")
    check(episode_is_single_sequence(ep), "scripted append-only episode failed the gate")
    check(all(c.ok for c in check_episode(ep)), "per-pair checks disagree with the gate")
    # tau2 fidelity: history opens with the greeting, tool results carry tool_call_id,
    # and the agent request is seeded
    msgs = cl.last_body["messages"]
    check(msgs[1] == {"role": "assistant", "content": GREETING, "tool_calls": None},
          "history does not open with tau2's greeting")
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    check(tool_msgs and tool_msgs[0].get("tool_call_id") == "c1", "tool_call_id not threaded")
    check(cl.last_body["seed"] == 1017, "agent request not seeded")

    # 2. reward is 0 when the DB never reaches gold
    ep2, _ = run_episode(ScriptedRuntime(customer_script=[("bye", True)]),
                         ScriptedClient([{"text": "no idea"}]), ids("ep-2"))
    check(ep2.reward == 0, "no gold mutation should score 0")

    # 3. empty assistant message: terminal, reward 0, counted, never retried
    ep3, st3 = run_episode(ScriptedRuntime(), ScriptedClient([{"text": "   "}]), ids("ep-3"))
    check(ep3.termination == "empty_message", f"got {ep3.termination}")
    check(ep3.reward == 0 and st3.empty_messages == 1, "empty message not counted")
    check(ep3.trains, "empty message should still train at reward 0")

    # 4. truncation is excluded from training
    ep4, _ = run_episode(ScriptedRuntime(),
                         ScriptedClient([{"text": "long", "finish_reason": "length"}]),
                         ids("ep-4"))
    check(ep4.termination == "length" and ep4.trains and ep4.reward == 0,
          "truncation must train at reward 0")

    # 5. max_steps, counted tau2's way: opening(1) + agent(2) + customer(3) + agent(4) ...
    ep5, _ = run_episode(ScriptedRuntime(customer_script=[("go on", False)] * 50),
                         ScriptedClient([{"text": "hm"}] * 50), ids("ep-5"),
                         Guards(max_steps=6))
    check(ep5.termination == "max_turns", f"got {ep5.termination}")
    check(ep5.extra["steps"] == 6, f"stopped at step {ep5.extra['steps']}, want 6")

    # 6. no-progress timeout, and it still trains at reward 0
    ep6, _ = run_episode(ScriptedRuntime(),
                         ScriptedClient([], raise_on=TimeoutError("slow")), ids("ep-6"))
    check(ep6.termination == "no_progress_timeout", f"got {ep6.termination}")
    check(ep6.trains and ep6.reward == 0, "timeout should train at reward 0")

    # 7. customer API failure: infrastructure, excluded, re-rolled
    ep7, st7 = run_episode(ScriptedRuntime(fail_customer=True),
                           ScriptedClient([{"text": "hello"}]), ids("ep-7"))
    check(ep7.termination == "customer_api_error", f"got {ep7.termination}")
    check(not ep7.trains and ep7.needs_reroll, "customer failure must re-roll, not train")
    check(st7.customer_retries == 3, f"expected 3 attempts, saw {st7.customer_retries}")

    # 8. a failing tool call is fed back as a tool result, as tau2 does, and 10 of them end
    #    the episode as too_many_errors
    ep8, st8 = run_episode(ScriptedRuntime(),
                           ScriptedClient([{"text": "", "tool_calls": [tc("nope", "{}")]}] * 20),
                           ids("ep-8"))
    check(ep8.termination == "too_many_errors", f"got {ep8.termination}")
    check(st8.tool_errors == 10 and ep8.trains, "tool errors not counted or not trained")
    check(any(isinstance(t, EnvTurn) and t.content.startswith("Error:") for t in ep8.turns),
          "tool error not returned to the agent")

    # 9. two tool calls in one turn both execute
    ep9, _ = run_episode(
        ScriptedRuntime(customer_script=[("bye", True)]),
        ScriptedClient([{"text": "", "tool_calls": [tc("get_order_details", "{}", "a"),
                                                     tc(i="b")]}, {"text": "done"}]),
        ids("ep-9"))
    check(ep9.reward == 1, "second tool call in a turn did not execute")
    check(len(ep9.assistant_turns[0].tool_calls or []) == 2, "tool_calls not recorded")

    # 10. malformed tool-call JSON is a policy failure, trained at 0
    ep10, _ = run_episode(ScriptedRuntime(),
                          ScriptedClient([{"text": "", "tool_calls": [tc(args="{bad")]}]),
                          ids("ep-10"))
    check(ep10.termination == "tool_parse_error" and ep10.trains, f"got {ep10.termination}")

    # 11. missing token IDs must fail loudly, never silently re-tokenize
    for bad in ({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]},
                {"prompt_token_ids": [1],
                 "choices": [{"message": {"content": "x"}, "token_ids": [2, 3],
                              "logprobs": {"content": [{"logprob": -0.1}]},
                              "finish_reason": "stop"}]}):
        try:
            parse_completion(bad)
            failures.append("bad completion did not raise")
        except VLLMError:
            pass

    # 13. goodbye loop: after the write, two closing-only customer messages end the episode
    #     as natural and the DB is scored, instead of looping to the step cap at reward 0
    loop = [("No, that's all. Thank you very much for your help! Have a nice day.", False),
            ("Thank you! I appreciate it. Goodbye!", False)] + [("Goodbye!", False)] * 40
    ep13, _ = run_episode(ScriptedRuntime(customer_script=loop),
                          ScriptedClient([{"text": "", "tool_calls": [tc()]},
                                          {"text": "Cancelled. Anything else?"}]
                                         + [{"text": "Goodbye! Take care!"}] * 40),
                          ids("ep-13"), Guards(max_steps=80))
    check(ep13.termination == "natural" and ep13.reward == 1,
          f"goodbye loop not closed: {ep13.termination} reward {ep13.reward}")
    check(ep13.extra["closed_by_backstop"] and ep13.extra["steps"] <= 8,
          f"backstop fired late or not at all: steps {ep13.extra['steps']}")

    # 14. a thank-you that carries a new request is NOT a close, so the episode continues
    ep14, _ = run_episode(
        ScriptedRuntime(customer_script=[("Thanks! That's all.", False),
                                         ("Thank you. Actually, can you also change my address?", False),
                                         ("Thanks, bye!", False), ("ok", True)]),
        ScriptedClient([{"text": "done"}] * 10), ids("ep-14"))
    check(not ep14.extra["closed_by_backstop"], "backstop fired across a new request")

    # 15. the classifier on real messages from the Sept 24 transcripts
    for text, want in [("Goodbye!", True), ("Thank you! You too! Take care!", True),
                       ("Thank you! Wishing you a great day as well. Goodbye!", True),
                       ("No, that's all. Thank you very much for your help! Have a nice day.", True),
                       ("Thanks!\n\n###STOP###", True),
                       ("Yes, I confirm. Please go ahead and cancel the order. Thank you!", False),
                       ("Thanks. Can you also check order #W4923227?", False),
                       ("Oh, yes, the order is #W4923227.", False), ("###STOP###", False)]:
        check(is_closing_only(text) == want, f"is_closing_only({text!r}) != {want}")
    check("###STOP###" in CLOSING_INSTRUCTION, "closing instruction must name the stop token")

    # 17. customer pricing: a priced model is charged from token counts, never $0 (the budget
    #     guard depends on it); an unpriced model falls back to litellm's own figure
    class _M:
        def __init__(self, usage=None, cost=None):
            self.usage, self.cost = usage, cost
    ds = CUSTOMER_MODEL
    check(abs(customer_cost(ds, _M({"prompt_tokens": 6400, "completion_tokens": 250}, cost=0.0))
              - (6400 * 0.09 + 250 * 0.18) / 1e6) < 1e-12, "priced customer cost wrong")
    check(customer_cost(ds, _M(None, cost=0.0)) > 0, "priced customer with no usage cost $0")
    check(customer_cost("gpt-4.1-nano", _M(None, cost=0.0123)) == 0.0123, "litellm fallback lost")

    # 16. a stop attached to a confirmation is ignored so the agent can act; the write then
    #     happens and a real goodbye stop ends the episode at reward 1
    ep16, _ = run_episode(
        ScriptedRuntime(customer_script=[
            ("Yes, I confirm. Please proceed with the cancellation. Thank you.###STOP###", True),
            ("Great, thank you! Goodbye! ###STOP###", True)]),
        ScriptedClient([{"text": "Please confirm (yes) to cancel."},
                        {"text": "", "tool_calls": [tc()]}, {"text": "Cancelled."}]),
        ids("ep-16"))
    check(ep16.reward == 1 and ep16.extra["stop_overrides"] == 1,
          f"premature stop not overridden: reward {ep16.reward} overrides {ep16.extra['stop_overrides']}")
    check(not any("###STOP###" in t.content for t in ep16.turns[:-1] if isinstance(t, EnvTurn)),
          "overridden stop token leaked to the agent")
    # ...and overrides are bounded: a customer that keeps stopping on confirmations still ends
    ep16b, _ = run_episode(
        ScriptedRuntime(customer_script=[("Yes, go ahead. ###STOP###", True)] * 10),
        ScriptedClient([{"text": "Shall I proceed?"}] * 10), ids("ep-16b"))
    check(ep16b.termination == "natural" and ep16b.extra["stop_overrides"] == MAX_STOP_OVERRIDES,
          f"override not bounded: {ep16b.extra['stop_overrides']}")
    for text, want in [("Yes, I want to proceed with the return. Please go ahead.###STOP###", True),
                       ("Yes, I confirm I want to proceed with the return of the laptop. ###STOP###", True),
                       ("No, that covers everything. Thank you. Goodbye! ###STOP###", False),
                       ("Yes, that's all, thank you! Goodbye ###STOP###", False),
                       ("I need to speak with a human agent. ###STOP###", False),
                       ("###STOP###", False), ("Yes, go ahead.", False)]:
        check(premature_stop(text) == want, f"premature_stop({text!r}) != {want}")

    # 12. batch runner: groups written whole, a group that cannot be refilled is dropped,
    #     spend lands in the ledger, and the budget refuses the next batch
    tmp = Path(tempfile.mkdtemp(prefix="driver-selftest-"))
    old_ledger = LEDGER_PATH
    LEDGER_PATH = tmp / "ledger.jsonl"
    try:
        def factory(task_id, seed):
            return ScriptedRuntime(task_id=task_id, customer_script=[("bye", True)],
                                   fail_customer=(task_id == "bad"), cost=0.001)
        m = {"batch_id": 3, "policy_version": "base-b7ea907", "model": "x",
             "tasks": ["t1", "t2", "bad"], "G": 4, "seed": 7, "out_dir": str(tmp / "rec"),
             "concurrency": 4, "max_rerolls": 1}
        s = run_batch(m, runtime_factory=factory, client=ScriptedClient([]))
        check(s["groups_kept"] == 2 and s["groups_dropped"] == 1, f"groups {s}")
        check(s["episodes_written"] == 8, f"wrote {s['episodes_written']}, want 8")
        check(spent_usd() > 0, "spend not recorded in the ledger")
        ep_back = read_episode(next((tmp / "rec").glob("*.json.gz")))
        check(ep_back.batch_id == 3, "record round-trip lost batch_id")
        record_spend(API_BUDGET_USD, "fill")
        try:
            run_batch(m, runtime_factory=factory, client=ScriptedClient([]))
            failures.append("budget did not refuse the batch")
        except BudgetExceeded:
            pass
    finally:
        LEDGER_PATH = old_ledger
        shutil.rmtree(tmp, ignore_errors=True)

    # 18. tool hints: only errors get one; swaps get the pairing hint
    check(tool_hint("modify_pending_order_items", "Error: new item id should be different") == SWAP_HINT,
          "swap error without the pairing hint")
    check(tool_hint("modify_pending_order_items", "{ok}") == "", "hint on a successful call")
    check(tool_hint("get_order_details", "Error: order not found") == LOOKUP_HINT, "no lookup hint")
    check(Guards(**{"tool_hints": True}).tool_hints, "tool_hints not settable from a manifest")

    # 19. communicate: tau2's substring rule, and the reward is DB x COMMUNICATE
    check(communicated(["Your refund is $1,234.50."], ["1234.50"]), "comma-stripped match missed")
    check(not communicated(["Refund issued."], ["1234.50"]), "unsaid info counted as said")
    check(communicated(["anything"], []), "empty communicate_info must pass")

    class _Says(ScriptedRuntime):
        def __init__(self, infos, **kw):
            super().__init__(**kw)
            self._infos = infos

        def communicate_info(self):
            return self._infos
    for infos, want in ((["done"], 1), (["tracking 123"], 0)):
        cl = ScriptedClient([{"text": "", "tool_calls": [tc()]}, {"text": "done"}])
        e, _ = run_episode(_Says(infos, customer_script=[("ok thanks", True)]), cl, ids())
        check(e.reward == want and e.extra["db_ok"] is True
              and e.extra["communicate_ok"] is bool(want), f"communicate {infos}: {e.reward} {e.extra}")

    # 20. personas: one per group, only from the manifest; a manifest without them draws the
    #     same episode seeds as before personas existed
    tmp = Path(tempfile.mkdtemp(prefix="driver-selftest-"))
    old_ledger = LEDGER_PATH
    LEDGER_PATH = tmp / "ledger.jsonl"
    seen: dict[str, set] = {}
    try:
        def pfactory(task_id, seed, persona=None):
            rt = ScriptedRuntime(task_id=task_id, customer_script=[("bye", True)])
            rt.persona = persona
            return rt
        base_m = {"batch_id": 4, "policy_version": "v", "model": "x", "tasks": ["t1", "t2", "t3"],
                  "G": 3, "seed": 11, "concurrency": 3}
        plain = run_batch(dict(base_m, out_dir=str(tmp / "a")), runtime_factory=pfactory,
                          client=ScriptedClient([]))
        withp = run_batch(dict(base_m, out_dir=str(tmp / "b"), personas={"terse": 1, "forthcoming": 1}),
                          runtime_factory=pfactory, client=ScriptedClient([]))
        check(plain["episodes_written"] == withp["episodes_written"] == 9, "persona batch lost episodes")
        sa = sorted((e.episode_id, e.seed) for e in map(read_episode, (tmp / "a").glob("*.json.gz")))
        sb = sorted((e.episode_id, e.seed) for e in map(read_episode, (tmp / "b").glob("*.json.gz")))
        check(sa == sb, "personas changed the episode seeds")
        for e in map(read_episode, (tmp / "b").glob("*.json.gz")):
            seen.setdefault(e.group_id, set()).add(e.extra.get("persona"))
        check(all(len(v) == 1 and None not in v for v in seen.values()), f"persona not per group: {seen}")
        check(all(e.extra.get("persona") is None for e in map(read_episode, (tmp / "a").glob("*.json.gz"))),
              "persona set without a manifest entry")

        # 21. customer mix: one customer per group, same episode seeds, both customers drawn
        def cfactory(task_id, seed, persona=None, customer=None):
            rt = pfactory(task_id, seed, persona)
            rt.customer_name = customer
            return rt
        mix = {"ds": {"model": "m1", "args": {"temperature": 1.0}, "weight": 1},
               "oa": {"model": "m2", "weight": 1}}
        withc = run_batch(dict(base_m, tasks=[f"t{i}" for i in range(12)], out_dir=str(tmp / "c"),
                               customers=mix), runtime_factory=cfactory, client=ScriptedClient([]))
        check(withc["episodes_written"] == 36, "customer batch lost episodes")
        cseen: dict[str, set] = {}
        for e in map(read_episode, (tmp / "c").glob("*.json.gz")):
            cseen.setdefault(e.group_id, set()).add(e.extra.get("customer"))
        check(all(len(v) == 1 and None not in v for v in cseen.values()), f"customer not per group: {cseen}")
        check(set().union(*cseen.values()) == {"ds", "oa"}, f"customer mix drew {cseen}")
        try:
            run_batch(dict(base_m, out_dir=str(tmp / "d"), customers={"x": {"weight": 1}}),
                      runtime_factory=cfactory, client=ScriptedClient([]))
            failures.append("customer without a model accepted")
        except ValueError:
            pass
    finally:
        LEDGER_PATH = old_ledger
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("SELFTEST FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("selftest ok (21 scenarios)")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "selftest":
        return _selftest()
    if cmd == "batch" and len(argv) > 2:
        manifest = json.loads(Path(argv[2]).read_text())
        t0 = time.time()
        summary = run_batch(manifest)
        summary["wall_s"] = round(time.time() - t0, 1)
        print(json.dumps(summary))
        return 0
    if cmd == "spend":
        print(f"${spent_usd():.4f} of ${API_BUDGET_USD:.2f}")
        return 0
    print(__doc__.strip().rsplit("CLI", 1)[-1].strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
