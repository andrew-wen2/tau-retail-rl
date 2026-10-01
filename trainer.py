"""Multi-turn GRPO trainer. Runs in .venv-train on the VM.

The trainer owns the run: it samples tasks, writes each batch's manifest, runs the rollout
driver (driver.py, in .venv-tau2) as a subprocess against the training vLLM server, reads
the records it writes, takes ONE optimizer step on them, writes the new LoRA adapter, swaps
it into the server, and checkpoints. There is no supervisor; the subprocess exiting is the
"drained" signal, so nothing in flight ever spans a policy version.

What the gradient is:
  * std-normalized group advantage A = (r - mean) / (std + 1e-4), population std. For
    binary rewards this is bounded by sqrt(G-1), which is the near-miss upweighting the
    pass^4 target relies on, so it is kept on purpose (against Dr. GRPO).
  * one scalar episode reward broadcast to every policy-generated token; customer and tool
    tokens are masked out by construction (records.assemble builds the mask from token IDs)
  * token-level normalization over the whole batch (DAPO), not per-sequence means
  * beta = 0, but KL-to-base is estimated and logged every step
  * one gradient step per rollout batch, so no PPO ratio or clip is needed for
    off-policyness. Truncated importance sampling corrects the vLLM-vs-trainer numerical
    drift instead: w = min(exp(logp_trainer - logp_vllm), C), detached. Both sides are RAW
    logprobs (serve-train.sh --logprobs-mode raw_logprobs vs a plain log_softmax here), so
    the ratio measures DeltaNet/bf16 drift and not sampling parameters.

Memory: the loss only needs logprobs at policy-generated positions, which are ~25% of an
episode. So the trainer runs the decoder once, gathers the hidden states at those positions,
and applies the LM head in checkpointed chunks. A full-sequence logits tensor at a ~250k
vocabulary would be the largest allocation in the step by far; this never builds it.

Checkpoint and resume (the VM is Spot): at every step boundary the adapter, optimizer,
scheduler, RNG state, sampler posteriors, step counter and last fully consumed batch_id are
written atomically. On resume, records from any batch past the checkpoint are deleted and
the batch is re-run under the restored policy, so a preemption costs at most one step.

CLI:
    trainer.py selftest                       advantages, masking, loss, TIS, checkpoint
                                              round-trip and bit-exact resume, on a tiny
                                              random model (CPU is fine)
    trainer.py probe                          load Qwen3.5-2B + LoRA, report the module
                                              names targeted, one fwd/bwd at --seq-len,
                                              peak memory (the GPU_MEM_UTIL assignment)
    trainer.py parity <records_dir>           zero-init-adapter logprob parity against the
                                              logprobs vLLM captured in the records
    trainer.py init-adapter <dir> [--random]  write an r=32 adapter (zero-init, or random B
                                              for the serving smoke test)
    trainer.py run --run <name> [...]         the loop; --resume continues a killed run
    trainer.py gate <name>                    did train reward go up over a run?   
    trainer.py direction <name> <records_dir> do successes gain logprob vs failures under
                                              the run's adapter? (use untrained-on episodes)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from records import Episode, assemble, auth_before_first_write, episode_is_single_sequence
    from records import iter_episodes, unrequested_writes
    from records import FAILURE_CLASSES, failure_class, gold_write_progress, graded_write_credit
    from records import rejected_writes, repeated_calls
except ImportError:  # imported as a package
    from .records import Episode, assemble, auth_before_first_write  # type: ignore
    from .records import episode_is_single_sequence, iter_episodes  # type: ignore
    from .records import unrequested_writes  # type: ignore
    from .records import FAILURE_CLASSES, failure_class, gold_write_progress, graded_write_credit  # type: ignore
    from .records import rejected_writes, repeated_calls  # type: ignore

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"                             # task sets, splits, profiles
TAU2_PYTHON = HERE / "baseline" / ".venv-tau2" / "bin" / "python"

# Plan requirement 3: attention and MLP projections plus DeltaNet's fused projections. Skip
# in_proj_a/b, which vLLM ignores. Matched as module-name suffixes inside the language model.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "out_proj"]


# ------------------------------------------------------------------------------ config


@dataclass
class Config:
    model: str = "Qwen/Qwen3.5-2B"
    served_base: str = "qwen3.5-2B"
    adapter_name: str = "policy"
    base_url: str = "http://localhost:8000/v1"
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    # optimizer
    lr: float = 1e-5
    warmup_steps: int = 5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    adv_eps: float = 1e-4
    tis_cap: float = 2.0
    # batching
    steps: int = 20
    G: int = 8
    episodes_per_step: int = 128
    tasks: list[str] = field(default_factory=list)   # fixed slice; empty = value-weighted
    sampler: str = "fixed"                             # fixed | value
    # value weighting: "pass4" = P(mixed) * 4E[p^3] (chases near-solved tasks, mean p ~0.78);
    # "mixed" = P(mixed) alone (centres on p~0.5, nearly every group carries signal). Sept 26.
    sampler_weight: str = "pass4"
    # value pool: "mixed" = only tasks the profile saw mixed; "all" = every train task, left to
    # the weight (profile-hard: 0/8 tasks still give a mixed group 41% of the time, 8/8 19%)
    sampler_pool: str = "mixed"                        # mixed | all | not_always (drop 8/8)
    # warm start: a fresh run whose policy starts from this adapter dir (the trainer-layout
    # copy, runs/<run>/adapters/<version>) with a fresh optimizer. Sept 26: p4-c from p4-b's
    # step-15 adapter.
    init_adapter: str = ""
    # Training reward = eval reward - this * [any unrequested successful write] (records.
    # unrequested_writes). 0 = pure DB reward. Metrics and val always report the eval reward.
    unrequested_penalty: float = 0.0
    # Sept 26, p4-e (the task-variety research round):
    # gated partial credit -- a FAILED episode whose successful writes are all gold writes earns
    # partial_credit * (gold writes done / gold writes); any non-gold write zeroes it
    # (records.gold_write_progress). Training reward only.
    partial_credit: float = 0.0
    # "gated" (p4-e..p4-j): any non-gold write zeroes the credit. "graded" (Sept 29): argument-
    # level closeness per gold write, unmatched writes subtract (records.graded_write_credit).
    credit_mode: str = "gated"
    # Sept 29: a SUCCESS loses mistake_penalty per rejected write call or non-gold successful
    # write (wrong-then-fixed), capped at mistake_cap, so it stays far above any failure (<=
    # partial_credit). Gives all-success groups whose attempts stumbled a gradient toward the
    # clean route. 0 = off.
    mistake_penalty: float = 0.0
    mistake_cap: float = 0.2
    # Sept 29 (RAFT): every episode loses repeat_penalty per tool call identical to an earlier one
    # (records.repeated_calls), capped at repeat_cap. With mistake_cap a success stays >= 0.6,
    # above any failure's partial credit. Training reward only. 0 = off.
    repeat_penalty: float = 0.0
    repeat_cap: float = 0.2
    # KL(pi||base) penalty in the loss (k3, per trained token, every sequence). 0 = monitor only.
    kl_coef: float = 0.0
    # Sept 30: what the penalty anchors to. "base" = the untrained model (every run to p4-o);
    # "init" = a frozen copy of init_adapter (soup4), loaded as a second adapter, so the penalty
    # starts at 0 and bounds the move away from the starting checkpoint instead of from base.
    # p4-m (4e-5) and p4-o (3e-5) drifted off soup4 in 2-4 steps with a 0.05 base penalty.
    # kl_to_base is still logged (every kl_every-th sequence) and drives the watcher's stop rule.
    kl_ref: str = "base"
    # Sept 30: skip the optimizer step when more than this fraction of the batch hit the per-turn
    # token cap (termination "length"). p4-o's step 0 drew 8 runaways from soup4 (p4-m/p4-n 0-2 on
    # the same draw) and its update started the drift. 0 = never skip.
    max_runaway_frac: float = 0.0
    max_tokens_per_turn: int = 2048                    # driver.Guards; the eval harness has none
    tool_hints_until: int = 0                          # driver tool-error hints for steps < this
    # targeted families (gen_targeted.py, ids tt_<family>_L<level>_<n>): start every family at
    # level 1, promote when its recent success passes promote_at, and multiply a family's
    # sampling weight by 1 + family_boost * (share of recent failures in the class it targets)
    family_adapt: bool = False
    family_boost: float = 3.0
    promote_at: float = 0.85
    # retire a task once it has retire_n observed attempts and posterior mean above retire_p
    retire_n: int = 0
    retire_p: float = 0.93
    concurrency: int = 64
    max_steps_per_episode: int = 80                    # training-only cap, see driver.Guards
    customer_model: str = "deepinfra/deepseek-ai/DeepSeek-V4-Flash"  # = driver.CUSTOMER_MODEL
    max_seq_len: int = 40960
    head_chunk: int = 1024
    kl_every: int = 4                                  # KL-to-base on every k-th sequence
    # Validation. 0 = off. Every val_every steps, at step 0 and at the
    # end, the 60 val tasks x val_G with the served adapter, never trained on, same seed every
    # pass so checkpoints are compared on identical customers and sampling seeds.
    tasks_file: str = "tb500_retail.json"              # or tb500_hard.json
    split_file: str = "split_tb500.json"               # split_hard.json pairs with tb500_hard.json
    profile_file: str = "profile.json"                 # the sampler's prior; customer- and task-specific
    val_repeat0: bool = False                          # second val pass at step 0: the noise floor
    val_every: int = 0
    val_G: int = 4
    seed: int = 300
    # Sept 28: branches of one experiment use different `seed`s (different task draws and
    # rollout seeds) but must be read on the SAME val customers and seeds; None keeps the old
    # derivation, which for seed 300 is valbig's 31000899.
    val_seed: int | None = None
    # Customer personas for TRAINING rollouts only ({driver.PERSONAS name: weight}); val never.
    personas: dict | None = None
    # Sept 29: a customer MIX for training rollouts only, one customer per group (driver
    # "customers": {name: {"model", "args", "weight"}}). None = customer_model at tau2's
    # temperature 0. Val always uses customer_model at temperature 0, so reads stay comparable.
    customers: dict | None = None
    # Sept 29, p4-l vet: gpt-5-nano role-flips into the agent and runs 4 of 32 episodes to the
    # step cap (DeepSeek 0 of 32; gpt-5.2 ended 909 of 912 eval episodes itself). Those zeros are
    # the customer's, so max_turns episodes get no advantage (their group is normalized over the
    # rest). Metrics and reward still count them.
    exclude_max_turns: bool = False
    grad_checkpointing: bool = True


# -------------------------------------------------------------------------- the gradient


def group_advantages(eps: list[Episode], eps_std: float = 1e-4,
                     rewards: dict[str, float] | None = None) -> tuple[dict[str, float], dict]:
    """A = (r - mean) / (std + eps) per group over its trainable members. Returns the
    advantage for every trainable episode in a non-degenerate group, and group stats.
    `rewards` (episode_id -> training reward) overrides eval_reward, for reward shaping.

    Degenerate groups (all rewards equal) carry zero advantage and are counted, since the
    fraction of them is the metric premise 2 says will kill the run."""
    groups: dict[str, list[Episode]] = {}
    for e in eps:
        if e.trains:
            groups.setdefault(e.group_id, []).append(e)
    adv: dict[str, float] = {}
    n_degenerate = n_all_success = 0
    for members in groups.values():
        rs = [float(rewards[e.episode_id] if rewards else e.eval_reward) for e in members]
        if len(rs) < 2:
            n_degenerate += 1
            continue
        mean = sum(rs) / len(rs)
        std = math.sqrt(sum((r - mean) ** 2 for r in rs) / len(rs))
        if std == 0.0:
            n_degenerate += 1
            n_all_success += mean >= 1.0
            continue
        for e, r in zip(members, rs):
            adv[e.episode_id] = (r - mean) / (std + eps_std)
    return adv, {"groups": len(groups), "degenerate": n_degenerate,
                 "degenerate_all_success": n_all_success,
                 "degenerate_frac": n_degenerate / len(groups) if groups else None}


def training_reward(e: Episode, gold_actions: list[dict], n_unrequested: int, cfg: Config) -> float:
    """The reward GRPO sees (p4-e). A success is 1. A failure earns partial_credit times the
    fraction of gold writes it did exactly, gated to 0 by any write that matches no gold write;
    then unrequested_penalty if it wrote to an order/user the task never asked about. Metrics and
    val report eval_reward, never this."""
    rep = min(cfg.repeat_cap, cfg.repeat_penalty * repeated_calls(e)) if cfg.repeat_penalty else 0.0
    if e.eval_reward:
        if not cfg.mistake_penalty:
            return 1.0 - rep
        n = rejected_writes(e) + gold_write_progress(e, gold_actions)[2]
        return 1.0 - min(cfg.mistake_cap, cfg.mistake_penalty * n) - rep
    r = -rep
    if cfg.partial_credit and cfg.credit_mode == "graded":
        r += cfg.partial_credit * graded_write_credit(e, gold_actions)
    elif cfg.partial_credit:
        matched, total, wrong = gold_write_progress(e, gold_actions)
        if total and not wrong:
            r += cfg.partial_credit * matched / total
    return r - cfg.unrequested_penalty * (n_unrequested > 0)


@dataclass
class TrainSeq:
    token_ids: torch.Tensor      # [L] long
    positions: torch.Tensor      # [n] hidden-state positions whose next token is trained
    targets: torch.Tensor        # [n] the generated token at position+1
    behavior_lp: torch.Tensor    # [n] vLLM's logprob for each target
    advantage: float
    single: bool


def build_sequences(eps: list[Episode], adv: dict[str, float],
                    max_len: int) -> tuple[list[TrainSeq], dict]:
    seqs: list[TrainSeq] = []
    n_too_long = n_single = n_eps = 0
    for e in eps:
        a = adv.get(e.episode_id)
        if a is None:
            continue
        n_eps += 1
        parts = assemble(e)
        single = len(parts) == 1 and len(e.assistant_turns) >= 1
        n_single += episode_is_single_sequence(e)
        for p in parts:
            if len(p.token_ids) > max_len:
                n_too_long += 1
                continue
            idx = [i for i, m in enumerate(p.loss_mask) if m]
            if not idx:
                continue
            seqs.append(TrainSeq(
                token_ids=torch.tensor(p.token_ids, dtype=torch.long),
                positions=torch.tensor([i - 1 for i in idx], dtype=torch.long),
                targets=torch.tensor([p.token_ids[i] for i in idx], dtype=torch.long),
                behavior_lp=torch.tensor([p.logprobs[i] for i in idx], dtype=torch.float32),
                advantage=a, single=single))
    return seqs, {"episodes_trained": n_eps, "sequences": len(seqs),
                  "single_sequence_rate": n_single / n_eps if n_eps else None,
                  "too_long": n_too_long}


# --------------------------------------------------------------------- model plumbing


def _decoder_and_head(model):
    """(decoder returning last_hidden_state, lm_head) for a plain or PEFT-wrapped CausalLM."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.get_decoder(), base.get_output_embeddings()


def _head_chunk(head, h: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = head(h).float()
    lse = torch.logsumexp(logits, dim=-1)
    lp = logits.gather(-1, tgt[:, None]).squeeze(-1) - lse
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1)
        ent = lse - (p * logits).sum(-1)
    return lp, ent


def token_logprobs(model, seq: TrainSeq, chunk: int, grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Logprob and entropy of each trained target under `model`, without ever materializing
    full-sequence logits: decoder once, gather the trained positions, head in chunks. With
    grad, each head chunk is checkpointed so its logits are recomputed in backward instead
    of stored."""
    dev = next(model.parameters()).device
    decoder, head = _decoder_and_head(model)
    ids = seq.token_ids.to(dev)[None]
    with torch.set_grad_enabled(grad):
        hidden = decoder(input_ids=ids).last_hidden_state[0]
        h = hidden.index_select(0, seq.positions.to(dev))
        tgt = seq.targets.to(dev)
        lps, ents = [], []
        for s in range(0, h.shape[0], chunk):
            hc, tc = h[s:s + chunk], tgt[s:s + chunk]
            if grad:
                lp, ent = checkpoint(_head_chunk, head, hc, tc, use_reentrant=False)
            else:
                lp, ent = _head_chunk(head, hc, tc)
            lps.append(lp)
            ents.append(ent)
    return torch.cat(lps), torch.cat(ents)


def load_policy(cfg: Config, device: str, adapter_dir: Path | None = None, random_b: bool = False):
    """Qwen3.5 + LoRA over the language model's projections. Returns the PEFT model."""
    from peft import LoraConfig, PeftModel, get_peft_model  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.bfloat16)
    base.to(device)
    if cfg.grad_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.config.use_cache = False
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
        if cfg.kl_ref == "init":
            # frozen reference for the KL penalty; loaded before the optimizer is built, and
            # is_trainable=False keeps its parameters out of it
            if not cfg.init_adapter:
                raise ValueError("kl_ref='init' needs init_adapter")
            model.load_adapter(str(HERE / cfg.init_adapter), adapter_name="ref", is_trainable=False)
            model.set_adapter("default")
    else:
        # Only modules inside the text decoder: a vision tower, if the checkpoint ships one,
        # is never served (--language-model-only) and must not be adapted.
        decoder_prefix = _decoder_prefix(base)
        names = [n for n, m in base.named_modules()
                 if isinstance(m, torch.nn.Linear) and n.startswith(decoder_prefix)
                 and n.rsplit(".", 1)[-1] in LORA_TARGETS]
        if not names:
            raise RuntimeError(f"no LoRA targets found under {decoder_prefix!r}")
        lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                          target_modules=names, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, lcfg)
        if random_b:
            g = torch.Generator().manual_seed(0)
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.data.copy_((torch.randn(p.shape, generator=g) * 1e-3).to(p.dtype))
    model.to(device)
    return model


def _decoder_prefix(base) -> str:
    dec = base.get_decoder()
    for n, m in base.named_modules():
        if m is dec:
            return n + "."
    return ""


def lora_param_count(model) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and ".ref." not in n)


@contextlib.contextmanager
def use_adapter(model, name: str):
    """Run under another loaded adapter (the frozen KL reference), then restore the policy.
    PEFT's set_adapter flips requires_grad to the active adapter, so restoring re-freezes the
    reference; callers wrap this in no_grad."""
    prev = model.active_adapter
    model.set_adapter(name)
    try:
        yield
    finally:
        model.set_adapter(prev)


def adapter_targets(model) -> dict[str, int]:
    out: dict[str, int] = {}
    for n, _ in model.named_modules():
        if n.endswith(".lora_A.default"):
            leaf = n[: -len(".lora_A.default")].rsplit(".", 1)[-1]
            out[leaf] = out.get(leaf, 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- one step


def train_step(model, opt, seqs: list[TrainSeq], cfg: Config, step: int) -> dict[str, Any]:
    """One optimizer step over the whole batch. Loss is summed over trained tokens and
    divided by the batch's total trained-token count (token-level normalization), so a long
    failing episode is not down-weighted per token."""
    model.train()
    total_tokens = sum(len(s.targets) for s in seqs)
    if total_tokens == 0:
        return {"skipped": "no trainable tokens"}
    opt.zero_grad(set_to_none=True)
    loss_sum = 0.0
    gaps: list[torch.Tensor] = []
    n_trunc = 0
    ent_sum = 0.0
    kl_sum, kl_tok = 0.0, 0
    ref_kl_sum, ref_kl_tok = 0.0, 0
    for i, s in enumerate(seqs):
        base_lp = None
        if cfg.kl_coef:
            # the penalty needs its reference on every sequence, computed before the policy's graph
            with torch.no_grad():
                if cfg.kl_ref == "init":
                    with use_adapter(model, "ref"):
                        base_lp, _ = token_logprobs(model, s, cfg.head_chunk, grad=False)
                else:
                    with model.disable_adapter():
                        base_lp, _ = token_logprobs(model, s, cfg.head_chunk, grad=False)
        lp, ent = token_logprobs(model, s, cfg.head_chunk, grad=True)
        blp = s.behavior_lp.to(lp.device)
        with torch.no_grad():
            diff = lp.detach() - blp
            w = torch.exp(diff).clamp(max=cfg.tis_cap)
            n_trunc += int((torch.exp(diff) > cfg.tis_cap).sum())
            gaps.append(diff.abs().float().cpu())
            ent_sum += float(ent.sum())
        loss = -(w * s.advantage * lp).sum() / total_tokens
        if base_lp is not None:
            d = base_lp.to(lp.device) - lp                 # k3 of KL(pi||ref), grad through lp
            k3 = torch.exp(d) - d - 1
            loss = loss + cfg.kl_coef * k3.sum() / total_tokens
            if cfg.kl_ref == "init":
                ref_kl_sum += float(k3.detach().sum())
                ref_kl_tok += len(d)
            else:
                kl_sum += float(k3.detach().sum())
                kl_tok += len(d)
        loss.backward()
        loss_sum += float(loss.detach())
        if (base_lp is None or cfg.kl_ref == "init") and cfg.kl_every and i % cfg.kl_every == 0:
            with torch.no_grad(), model.disable_adapter():
                blp_base, _ = token_logprobs(model, s, cfg.head_chunk, grad=False)
                d = blp_base - lp.detach()       # log(pi_base/pi); k3 estimator of KL(pi||base)
                kl_sum += float((torch.exp(d) - d - 1).sum())
                kl_tok += len(d)
    params = [p for p in model.parameters() if p.requires_grad]
    gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
    lr = cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
    for g in opt.param_groups:
        g["lr"] = lr
    opt.step()
    g_all = torch.cat(gaps) if gaps else torch.zeros(1)
    return {
        "loss": loss_sum, "grad_norm": gnorm, "lr": lr, "trained_tokens": total_tokens,
        "lp_gap_median": float(g_all.median()), "lp_gap_p99": float(torch.quantile(g_all, 0.99))
        if g_all.numel() > 1 else float(g_all.max()),
        "tis_truncated_frac": n_trunc / total_tokens,
        "entropy": ent_sum / total_tokens,
        "kl_to_base": kl_sum / kl_tok if kl_tok else None,
        "kl_to_ref": ref_kl_sum / ref_kl_tok if ref_kl_tok else None,
    }


# ------------------------------------------------------------------------------ sampler


def _beta_moment(a: float, b: float, k: int) -> float:
    """E[p^k] for p ~ Beta(a, b)."""
    r = 1.0
    for i in range(k):
        r *= (a + i) / (a + b + i)
    return r


def fit_beta_prior(counts: list[tuple[int, int]]) -> tuple[float, float]:
    """Empirical-Bayes Beta prior for per-task success rates, by beta-binomial method of
    moments over (successes, trials). Clamped: a degenerate fit falls back to Beta(1, 1)."""
    rows = [(s, n) for s, n in counts if n > 1]
    if len(rows) < 10:
        return 1.0, 1.0
    n = sum(n for _, n in rows) / len(rows)
    ps = [s / k for s, k in rows]
    m = sum(ps) / len(ps)
    v = sum((p - m) ** 2 for p in ps) / (len(ps) - 1)
    if not 0 < m < 1:
        return 1.0, 1.0
    rho = (v * n / (m * (1 - m)) - 1) / (n - 1)
    if not 0 < rho < 1:
        return 1.0, 1.0
    ab = 1 / rho - 1
    return max(0.05, m * ab), max(0.05, (1 - m) * ab)


# gen_targeted.py families and the failure class each one targets (records.failure_class)
FAMILY_TARGETS = {"items_addr": "missed_write", "decoy_first": "wrong_target",
                  "keep_options": "wrong_args", "pairing": "wrong_args"}


def family_of(task_id: str) -> tuple[str | None, int]:
    """("items_addr", 2) for "tt_items_addr_L2_007"; (None, 0) for any other task."""
    if not task_id.startswith("tt_"):
        return None, 0
    head, lv, _ = task_id[3:].rsplit("_", 2)
    return head, int(lv[1:])


class Sampler:
    """Task selection. `fixed` returns the same slice every step (the dev slice, so
    mean train reward is comparable across steps). `value` samples in proportion to the
    expected pass^4 gain per rollout, w = P(mixed group at G) * 4 E[p^3] / G, floored so no
    task is starved.

    Revised Sept 25 after the first value-sampled run drew 75% degenerate groups. Two fixes:
      * The prior is fitted to the profile (empirical Bayes), not Beta(1, 1). Task difficulty
        here is U-shaped (136 of 358 always solved, 51 never); under a flat prior an 8/8 task's
        posterior mean is 0.90, exactly where 4p^3 * P(mixed) peaks, so the sampler preferred
        the tasks that can never produce a mixed group. The fitted prior puts it near 0.97.
      * P(mixed) and E[p^3] are exact under the posterior instead of plugging in its mean, and
        the pool is the tasks that were mixed in the profile (1..n-1 successes). Offline over
        the 358 with the posterior as truth: degenerate groups 62% -> 24% and pass^4 signal
        per rollout +54% against the old weighting.

    Sept 26: `sampler_weight="mixed"` drops the 4E[p^3] factor. Under it step 0 of the first
    full run drew tasks at profile mean p 0.78 (pool 0.63) and lost 8 of 32 groups to
    all-success. GRPO normalizes each group by its own std, so easy groups earn no larger
    update, and the eval tasks are different tasks: what transfers is the skills the agent
    fails at, which live in the lower-p tasks the pass^4 weight almost never drew."""

    def __init__(self, cfg: Config, pool: list[str], prior: dict[str, dict] | None = None):
        self.cfg = cfg
        prior = prior or {}
        counts = {t: (prior[t]["successes"], prior[t]["trials"]) for t in pool if t in prior}
        self.a0, self.b0 = fit_beta_prior(list(counts.values())) if counts else (1.0, 1.0)
        # Tasks the profile saw all-success or all-fail almost never give a mixed group; the
        # floor would still spend rollouts on them. Without a profile, everything is eligible.
        mixed = [t for t in pool if t in counts and 0 < counts[t][0] < counts[t][1]]
        if cfg.sampler_pool == "not_always":
            self.pool = [t for t in pool if t not in counts or counts[t][0] < counts[t][1]]
        else:
            self.pool = mixed if mixed and cfg.sampler_pool == "mixed" else list(pool)
        self.ab: dict[str, list[float]] = {}
        for t in self.pool:
            s, n = counts.get(t, (0, 0))
            self.ab[t] = [self.a0 + s, self.b0 + n - s]
        self.rng = random.Random(cfg.seed)
        # adaptive targeted families (p4-e)
        self.level = {f: 1 for f in {family_of(t)[0] for t in self.pool} if f}
        self.fail_share = {c: 0.0 for c in FAILURE_CLASSES}
        self.retired: set[str] = set()

    def mean(self, t: str) -> float:
        a, b = self.ab[t]
        return a / (a + b)

    def weight(self, t: str, G: int) -> float:
        a, b = self.ab[t]
        p_mixed = 1.0 - _beta_moment(a, b, G) - _beta_moment(b, a, G)
        if self.cfg.sampler_weight == "mixed":
            return p_mixed
        return p_mixed * 4.0 * _beta_moment(a, b, 3) / G

    def next(self) -> tuple[list[str], int]:
        G = self.cfg.G
        if self.cfg.sampler == "fixed":
            return list(self.cfg.tasks), G
        n = max(1, self.cfg.episodes_per_step // G)
        w = {t: self.weight(t, G) * self.family_mult(t) for t in self.eligible()}
        floor = 0.02 * max(w.values())
        cand = {t: max(v, floor) for t, v in w.items()}
        chosen = []
        for _ in range(min(n, len(cand))):
            tot = sum(cand.values())
            r = self.rng.random() * tot
            for t, v in cand.items():
                r -= v
                if r <= 0:
                    chosen.append(t)
                    del cand[t]
                    break
        return chosen, G

    def update(self, eps: list[Episode], classes: dict[str, str | None] | None = None) -> dict:
        """Posterior update; with family_adapt also the failure-class shares, family promotion
        and retirement. `classes`: episode_id -> records.failure_class. Returns what changed."""
        for e in eps:
            if e.trains and e.task_id in self.ab:
                self.ab[e.task_id][0 if e.eval_reward else 1] += 1.0
        out: dict[str, Any] = {}
        if not self.cfg.family_adapt:
            return out
        if classes:
            fails = [c for c in classes.values() if c]
            if fails:
                for c in self.fail_share:           # EMA over steps
                    self.fail_share[c] = 0.5 * self.fail_share[c] + 0.5 * fails.count(c) / len(fails)
        by_fam: dict[str, list[int]] = {}
        for e in eps:
            f, lv = family_of(e.task_id)
            if f and e.trains and lv == self.level.get(f):
                by_fam.setdefault(f, []).append(e.eval_reward)
        for f, rs in by_fam.items():
            if len(rs) >= 16 and sum(rs) / len(rs) >= self.cfg.promote_at and self.level[f] < 3:
                self.level[f] += 1
                out.setdefault("promoted", []).append(f"{f}->L{self.level[f]}")
        if self.cfg.retire_n:
            for t, (a, b) in self.ab.items():
                if (t not in self.retired and a + b - self.a0 - self.b0 >= self.cfg.retire_n
                        and a / (a + b) > self.cfg.retire_p):
                    self.retired.add(t)
                    out["retired"] = out.get("retired", 0) + 1
        return out

    def eligible(self) -> list[str]:
        out = []
        for t in self.pool:
            if t in self.retired:
                continue
            f, lv = family_of(t)
            if f and self.cfg.family_adapt and lv > self.level.get(f, 1):
                continue
            out.append(t)
        return out

    def family_mult(self, t: str) -> float:
        f, lv = family_of(t)
        if not (f and self.cfg.family_adapt):
            return 1.0
        # the current level is where the difficulty is: earlier levels stay in at half weight
        at_level = 1.0 if lv == self.level.get(f, 1) else 0.5
        return at_level * (1.0 + self.cfg.family_boost * self.fail_share.get(FAMILY_TARGETS[f], 0.0))

    def state(self) -> dict:
        return {"ab": self.ab, "rng": self.rng.getstate(), "level": self.level,
                "fail_share": self.fail_share, "retired": sorted(self.retired)}

    def load(self, st: dict) -> None:
        # merge, not replace: a resume may widen the pool (sampler_pool "mixed" -> "all"), and
        # tasks new to it keep their profile posterior
        self.ab.update({k: list(v) for k, v in st["ab"].items()})
        s = st["rng"]
        self.rng.setstate((s[0], tuple(s[1]), s[2]))
        self.level.update(st.get("level", {}))
        self.fail_share.update(st.get("fail_share", {}))
        self.retired |= set(st.get("retired", []))


# ------------------------------------------------------------------ adapters and serving


# The trainer loads the text-only class (Qwen3_5ForCausalLM), whose decoder is `model.layers`.
# vLLM serves the checkpoint's own class, Qwen3_5ForConditionalGeneration, and maps LoRA
# names through its HF->vLLM mapper, which only knows the VL layout (`model.language_model.`
# -> `language_model.model.`). Adapter keys in the text-only layout match nothing, and vLLM
# then loads the adapter "successfully" while applying none of it: found Sept 23, when a
# random non-zero adapter returned the base model's logprobs to 4 decimals.
SERVE_KEY_PREFIX = ("base_model.model.model.layers.",
                    "base_model.model.model.language_model.layers.")


def export_for_vllm(src: Path, dst: Path) -> None:
    """Write the served copy of an adapter: same weights, VL-layout key names."""
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    old, new = SERVE_KEY_PREFIX
    tensors = load_file(str(src / "adapter_model.safetensors"))
    renamed = {(new + k[len(old):]) if k.startswith(old) else k: v for k, v in tensors.items()}
    if not any(k.startswith(new) for k in renamed):
        raise RuntimeError(f"no adapter keys under {old!r}; the served copy would be a no-op")
    dst.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(dst / "adapter_model.safetensors"))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")


def save_adapter(model, run_dir: Path, step: int) -> tuple[str, Path]:
    """Save to a versioned directory named {step:03d}-{sha12 of the weights}. A step index
    alone can repeat after a resume; the content hash cannot. The directory holds the
    trainer's copy (resume loads it) and `serve/`, the copy vLLM loads."""
    tmp = run_dir / "adapters" / f".tmp-{step:03d}"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(str(tmp), selected_adapters=["default"])
    export_for_vllm(tmp, tmp / "serve")
    h = hashlib.sha256((tmp / "adapter_model.safetensors").read_bytes()).hexdigest()[:12]
    version = f"{step:03d}-{h}"
    final = run_dir / "adapters" / version
    shutil.rmtree(final, ignore_errors=True)
    fsync_tree(tmp)
    os.replace(tmp, final)
    return version, final


def _post(url: str, body: dict, timeout: float = 120.0) -> tuple[int, str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def served_name(cfg: Config, version: str) -> str:
    return f"{cfg.adapter_name}-{version}"


def swap_adapter(cfg: Config, path: Path, version: str) -> str:
    """Serve this adapter version under a name used for no other version, then retire every
    other `{adapter_name}-*` adapter. Returns the name rollouts must request.

    Why not one reused name: through Sept 24 the loop unloaded and reloaded a single name
    ("policy") each step, and the vLLM-trainer logprob gap grew with training in every run,
    then fell back to its step-0 level the moment the server restarted at the same adapter
    size. Loading under a fresh name each step removes any cache keyed on the name. New is
    loaded before old is unloaded (max-cpu-loras 2), so no request ever finds nothing."""
    root = cfg.base_url.rsplit("/v1", 1)[0]
    name = served_name(cfg, version)
    code, msg = _post(f"{root}/v1/load_lora_adapter",
                      {"lora_name": name, "lora_path": str(path / "serve")})
    if code != 200 and "already" not in msg.lower():
        raise RuntimeError(f"load_lora_adapter failed: HTTP {code} {msg}")
    with urllib.request.urlopen(f"{root}/v1/models", timeout=60) as r:
        loaded = [m["id"] for m in json.loads(r.read())["data"]]
    for old in loaded:
        if old != name and (old == cfg.adapter_name or old.startswith(cfg.adapter_name + "-")):
            _post(f"{root}/v1/unload_lora_adapter", {"lora_name": old})
    return name


# ---------------------------------------------------------------------- checkpointing


def fsync_tree(path: Path) -> None:
    """Flush a file, or every file under a directory, to disk. A rename is not durable on its
    own: a Spot preemption right after os.replace can leave the new name pointing at zeros."""
    for f in ([path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]):
        fd = os.open(f, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save_checkpoint(run_dir: Path, state: dict) -> None:
    ck = run_dir / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    tmp = ck / ".tmp-state.pt"
    torch.save(state, tmp)
    fsync_tree(tmp)
    os.replace(tmp, ck / "state.pt")


def load_checkpoint(run_dir: Path) -> dict | None:
    p = run_dir / "ckpt" / "state.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def rng_state() -> dict:
    st = {"py": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["py"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


# ------------------------------------------------------------------------------ the loop


def pass_k(counts: list[tuple[int, int]], k: int) -> float | None:
    """pass^k = mean over tasks of C(c, k) / C(n, k), for (c successes, n trials) per task."""
    rows = [math.comb(c, k) / math.comb(n, k) for c, n in counts if n >= k]
    return sum(rows) / len(rows) if rows else None


def run_val(cfg: Config, run_dir: Path, step: int, version: str, serving: str,
            tasks: list[str], repeat: int = 0,
            gold: dict[str, list] | None = None) -> dict | None:
    """One validation pass with the currently served adapter. Idempotent across resumes: a
    step already in val.jsonl is not re-run."""
    log = run_dir / "val.jsonl"
    if log.exists() and any((json.loads(l).get("step"), json.loads(l).get("repeat", 0)) == (step, repeat)
                            for l in log.read_text().splitlines() if l.strip()):
        return None
    t0 = time.time()
    out_dir = run_dir / "val" / (f"v{step:04d}" + (f"r{repeat}" if repeat else ""))
    shutil.rmtree(out_dir, ignore_errors=True)
    manifest = {
        "batch_id": 100000 + step, "policy_version": version, "model": serving,
        "tasks": tasks, "G": cfg.val_G,
        "seed": cfg.val_seed if cfg.val_seed is not None else cfg.seed * 100003 + 999999,
        "out_dir": str(out_dir), "concurrency": cfg.concurrency,
        "customer_model": cfg.customer_model, "base_url": cfg.base_url,
        "guards": {"max_steps": cfg.max_steps_per_episode,
                   "max_tokens_per_turn": cfg.max_tokens_per_turn}, "tasks_file": cfg.tasks_file,
    }
    mpath = run_dir / "manifests" / (f"val{step:04d}" + (f"r{repeat}" if repeat else "") + ".json")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=1))
    proc = subprocess.run([str(TAU2_PYTHON), str(HERE / "driver.py"), "batch", str(mpath)],
                          capture_output=True, text=True, cwd=HERE)
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"driver failed on val at step {step}")
    roll = json.loads(proc.stdout.strip().splitlines()[-1])
    by_task: dict[str, list[int]] = {}
    val_eps = [e for e in iter_episodes(out_dir) if not e.needs_reroll]
    for e in val_eps:
        by_task.setdefault(e.task_id, []).append(e.eval_reward)
    natural = [e for e in val_eps if e.termination == "natural"]
    counts = [(sum(v), len(v)) for v in by_task.values()]
    rec = {"step": step, "repeat": repeat, "policy_version": version, "tasks": len(counts),
           "episodes": sum(n for _, n in counts),
           "pass1": pass_k(counts, 1), "pass2": pass_k(counts, 2), "pass4": pass_k(counts, 4),
           "terminations": roll.get("terminations"), "customer_usd": roll.get("customer_usd"),
           # Sept 26: the val gain through step 15 came from fewer runaway turns while success
           # on naturally-ended episodes fell; report both, and the unrequested-write rate
           "natural_success": sum(e.eval_reward for e in natural) / len(natural) if natural else None,
           "unrequested_write_frac": (sum(unrequested_writes(e, gold[e.task_id]) > 0 for e in val_eps)
                                      / len(val_eps)) if gold and val_eps else None,
           "t_val_s": round(time.time() - t0, 1)}
    with log.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print(json.dumps({"val": rec}), flush=True)
    return rec


def run(cfg: Config, run_name: str, resume: bool, device: str = "cuda") -> int:
    run_dir = HERE / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "metrics.jsonl"
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    tasks_all = json.loads((TASKS_DIR / cfg.split_file).read_text())
    gold = {t["id"]: t["evaluation_criteria"]["actions"]
            for t in json.loads((TASKS_DIR / cfg.tasks_file).read_text())}
    prior_path = TASKS_DIR / cfg.profile_file
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    sampler = Sampler(cfg, tasks_all["train"], prior)

    state = load_checkpoint(run_dir) if resume else None
    if resume and state is None:
        print("no checkpoint to resume; starting fresh", file=sys.stderr)
    if state is None and (run_dir / "ckpt" / "state.pt").exists():
        raise SystemExit(f"{run_dir} already has a checkpoint; pass --resume or a new --run")

    if state is None:
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
        model = load_policy(cfg, device,
                            adapter_dir=(HERE / cfg.init_adapter) if cfg.init_adapter else None)
        start, last_batch = 0, -1
        version, adir = save_adapter(model, run_dir, 0)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    else:
        start, last_batch = state["step"], state["last_batch_id"]
        version = state["policy_version"]
        adir = run_dir / "adapters" / version
        model = load_policy(cfg, device, adapter_dir=adir)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
        opt.load_state_dict(state["optimizer"])
        sampler.load(state["sampler"])
        set_rng_state(state["rng"])
        # Anything written after the checkpoint came from a batch that never trained.
        for d in (run_dir / "records").glob("b*"):
            if int(d.name[1:]) > last_batch:
                shutil.rmtree(d)
        print(f"resumed at step {start} (policy {version}, last batch {last_batch})")

    print(f"trainable LoRA params {lora_param_count(model):,}  targets {adapter_targets(model)}")
    serving = swap_adapter(cfg, adir, version)

    for step in range(start, cfg.steps):
        if cfg.val_every and step % cfg.val_every == 0:
            run_val(cfg, run_dir, step, version, serving, tasks_all["val"], gold=gold)
            if step == 0 and cfg.val_repeat0:
                run_val(cfg, run_dir, 0, version, serving, tasks_all["val"], repeat=1, gold=gold)
        t0 = time.time()
        tasks, G = sampler.next()
        batch_id = step
        out_dir = run_dir / "records" / f"b{batch_id:04d}"
        shutil.rmtree(out_dir, ignore_errors=True)
        manifest = {
            "batch_id": batch_id, "policy_version": version, "model": serving,
            "tasks": tasks, "G": G, "seed": cfg.seed * 100003 + batch_id,
            "out_dir": str(out_dir), "concurrency": cfg.concurrency,
            "customer_model": cfg.customer_model, "base_url": cfg.base_url,
            "guards": {"max_steps": cfg.max_steps_per_episode,
                       "max_tokens_per_turn": cfg.max_tokens_per_turn,
                       "tool_hints": step < cfg.tool_hints_until}, "tasks_file": cfg.tasks_file,
        }
        if cfg.personas:
            manifest["personas"] = cfg.personas
        if cfg.customers:
            manifest["customers"] = cfg.customers
        mpath = run_dir / "manifests" / f"b{batch_id:04d}.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(manifest, indent=1))
        proc = subprocess.run([str(TAU2_PYTHON), str(HERE / "driver.py"), "batch", str(mpath)],
                              capture_output=True, text=True, cwd=HERE)
        if proc.returncode != 0:
            print(proc.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"driver failed on batch {batch_id}")
        roll = json.loads(proc.stdout.strip().splitlines()[-1])
        t_roll = time.time() - t0
        if roll.get("aborted_on_budget"):
            raise SystemExit(f"API budget reached during batch {batch_id}: {roll}")

        eps = [e for e in iter_episodes(out_dir)]
        unreq = {e.episode_id: unrequested_writes(e, gold[e.task_id]) for e in eps}
        classes = {e.episode_id: failure_class(e, gold[e.task_id]) for e in eps if not e.needs_reroll}
        shaped = None
        if cfg.unrequested_penalty or cfg.partial_credit or cfg.mistake_penalty or cfg.repeat_penalty:
            shaped = {e.episode_id: training_reward(e, gold[e.task_id], unreq[e.episode_id], cfg)
                      for e in eps}
        adv_eps = [e for e in eps if not (cfg.exclude_max_turns and e.termination == "max_turns")]
        adv, gstats = group_advantages(adv_eps, cfg.adv_eps, shaped)
        seqs, sstats = build_sequences(eps, adv, cfg.max_seq_len)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        n_runaway = sum(e.termination == "length" for e in eps)
        if cfg.max_runaway_frac and eps and n_runaway / len(eps) > cfg.max_runaway_frac:
            tstats = {"skipped": f"{n_runaway} of {len(eps)} episodes hit the token cap"}
        else:
            tstats = train_step(model, opt, seqs, cfg, step)
        t_train = time.time() - t1

        sampled = sampler.update(eps, classes)
        new_version, new_dir = save_adapter(model, run_dir, step + 1)
        serving = swap_adapter(cfg, new_dir, new_version)
        save_checkpoint(run_dir, {
            "step": step + 1, "last_batch_id": batch_id, "policy_version": new_version,
            "optimizer": opt.state_dict(), "sampler": sampler.state(), "rng": rng_state(),
            "config": asdict(cfg)})
        version = new_version

        trainable = [e for e in eps if e.trains]
        # Reward over EVERY policy-attributable episode, so an episode class that leaves the
        # training pool cannot move the curve by leaving (the step-8 artifact of Sept 23).
        scored = [e for e in eps if not e.needs_reroll]
        auth = [auth_before_first_write(e) for e in trainable]
        auth = [a for a in auth if a is not None]
        rec = {
            "step": step, "batch_id": batch_id, "policy_version_rollout": manifest["policy_version"],
            "mean_reward": (sum(e.eval_reward for e in scored) / len(scored)) if scored else None,
            "length_truncated": sum(e.termination == "length" for e in eps),
            "episodes": len(eps), **gstats, **sstats, **tstats,
            "entropy_proxy_rollout": roll.get("entropy_proxy"),
            "terminations": roll.get("terminations"),
            "auth_before_first_write": sum(auth) / len(auth) if auth else None,
            "unrequested_write_frac": sum(unreq[e.episode_id] > 0 for e in eps) / len(eps)
            if eps else None,
            "mean_turns": (sum(e.turn_count for e in eps) / len(eps)) if eps else None,
            "failure_classes": dict(Counter(c for c in classes.values() if c)),
            # right DB, but a required fact never said (the reward counts it a failure)
            "stumbled_success_frac": (sum(rejected_writes(e) > 0 for e in eps if e.eval_reward)
                                      / max(1, sum(e.eval_reward for e in eps))),
            "unsaid_frac": sum(e.extra.get("db_ok") is True and e.extra.get("communicate_ok") is False
                               for e in eps) / len(eps) if eps else None,
            "persona_reward": {p: round(sum(e.eval_reward for e in eps if e.extra.get("persona") == p)
                                        / n, 3) for p, n in Counter(e.extra.get("persona")
                                                                   for e in eps).items() if p} or None,
            "customer_reward": {c: round(sum(e.eval_reward for e in eps if e.extra.get("customer") == c)
                                         / n, 3) for c, n in Counter(e.extra.get("customer")
                                                                    for e in eps).items() if c} or None,
            "excluded_max_turns": len(eps) - len(adv_eps),
            "repeated_calls_mean": (sum(repeated_calls(e) for e in eps) / len(eps)) if eps else None,
            "shaped_mean": (sum(shaped.values()) / len(shaped)) if shaped else None,
            "targeted_frac": sum(e.task_id.startswith("tt_") for e in eps) / len(eps) if eps else None,
            "family_level": dict(sampler.level), "sampler_changes": sampled or None,
            "tool_hints": manifest["guards"]["tool_hints"],
            "customer_usd": roll.get("customer_usd"), "ledger_usd": roll.get("ledger_usd"),
            "t_rollout_s": round(t_roll, 1), "t_train_s": round(t_train, 1),
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 2**30)
            if torch.cuda.is_available() else None,
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(json.dumps({k: rec.get(k) for k in ("step", "mean_reward", "degenerate_frac",
                                               "loss", "grad_norm", "lp_gap_median",
                                               "kl_to_base", "t_rollout_s", "t_train_s",
                                               "ledger_usd")}), flush=True)
    if cfg.val_every:
        run_val(cfg, run_dir, cfg.steps, version, serving, tasks_all["val"], gold=gold)
    return 0


# ------------------------------------------------------------------------ diagnostics


def probe(cfg: Config, seq_len: int, device: str = "cuda") -> int:
    model = load_policy(cfg, device)
    print("LoRA params", f"{lora_param_count(model):,}")
    print("targets", adapter_targets(model))
    L = seq_len
    ids = torch.randint(0, 150000, (L,))
    n = L // 4
    pos = torch.sort(torch.randperm(L - 1)[:n] ).values
    s = TrainSeq(ids, pos, ids[pos + 1], torch.full((n,), -1.0), 1.0, True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = train_step(model, opt, [s], cfg, 0)
    torch.cuda.synchronize()
    print(f"seq_len {L}: fwd+bwd+step {time.time()-t:.1f}s  "
          f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB  "
          f"reserved {torch.cuda.max_memory_reserved()/2**30:.1f} GiB  loss {out['loss']:.4f}")
    return 0


def parity(cfg: Config, rec_dir: Path, limit: int, device: str = "cuda") -> int:
    """Zero-init adapter vs the logprobs vLLM captured. Validates tokenization, template
    alignment, masking and the trainer's logprob math. It does NOT validate vLLM's LoRA
    path (a zero adapter is the base model); that needs a non-zero adapter on the server."""
    model = load_policy(cfg, device)
    model.eval()
    eps = [e for e in iter_episodes(rec_dir)][:limit]
    diffs = []
    for e in eps:
        for p in assemble(e):
            idx = [i for i, m in enumerate(p.loss_mask) if m]
            s = TrainSeq(torch.tensor(p.token_ids), torch.tensor([i - 1 for i in idx]),
                         torch.tensor([p.token_ids[i] for i in idx]),
                         torch.tensor([p.logprobs[i] for i in idx]), 0.0, True)
            with torch.no_grad():
                lp, _ = token_logprobs(model, s, cfg.head_chunk, grad=False)
            diffs.append((lp.float().cpu() - s.behavior_lp).abs())
    d = torch.cat(diffs)
    q = lambda x: float(torch.quantile(d, x))  # noqa: E731
    ok = q(0.5) < 0.01 and q(0.99) < 0.1
    print(json.dumps({"episodes": len(eps), "tokens": int(d.numel()), "median": q(0.5),
                      "p90": q(0.9), "p99": q(0.99), "max": float(d.max()),
                      "pass": ok}))
    return 0 if ok else 1


def direction(cfg: Config, run_name: str, rec_dir: Path, device: str = "cuda") -> int:
    """Is training pushing the right way? Score episodes under the base model and under the
    run's latest adapter. Within each mixed group, successes should gain agent-token logprob
    relative to failures. Use episodes the run never trained on (e.g. a base-policy batch on
    the same tasks), or the check only shows the update fit its own batch."""
    run_dir = HERE / "runs" / run_name
    st = load_checkpoint(run_dir)
    if st is None:
        raise SystemExit(f"{run_dir} has no checkpoint")
    model = load_policy(cfg, device, adapter_dir=run_dir / "adapters" / st["policy_version"])
    model.eval()
    eps = list(iter_episodes(rec_dir))
    adv, _ = group_advantages(eps)

    def mean_lp(e: Episode) -> float:
        tot, n = 0.0, 0
        for p in assemble(e):
            idx = [i for i, m in enumerate(p.loss_mask) if m]
            if not idx:
                continue
            s = TrainSeq(torch.tensor(p.token_ids), torch.tensor([i - 1 for i in idx]),
                         torch.tensor([p.token_ids[i] for i in idx]),
                         torch.tensor([p.logprobs[i] for i in idx]), 0.0, True)
            with torch.no_grad():
                lp, _ = token_logprobs(model, s, cfg.head_chunk, grad=False)
            tot += float(lp.float().sum()); n += len(idx)
        return tot / max(n, 1)

    delta: dict[str, float] = {}
    for e in eps:
        if e.episode_id not in adv:
            continue
        trained = mean_lp(e)
        with model.disable_adapter():
            base = mean_lp(e)
        delta[e.episode_id] = trained - base
    groups: dict[str, list[Episode]] = {}
    for e in eps:
        if e.episode_id in delta:
            groups.setdefault(e.group_id, []).append(e)
    per_group = []
    for g, members in groups.items():
        s = [delta[e.episode_id] for e in members if e.eval_reward == 1]
        f = [delta[e.episode_id] for e in members if e.eval_reward == 0]
        per_group.append((members[0].task_id, sum(s) / len(s) - sum(f) / len(f), len(s), len(f)))
    w = sum(adv[k] * d for k, d in delta.items()) / sum(abs(adv[k]) for k in delta)
    right = sum(d > 0 for _, d, _, _ in per_group)
    for t, d, ns, nf in sorted(per_group, key=lambda x: -x[1]):
        print(f"  {t}  success-minus-failure dlogp/token {d:+.5f}   ({ns} succ, {nf} fail)")
    print(json.dumps({"policy": st["policy_version"], "episodes": len(delta),
                      "mixed_groups": len(per_group), "groups_right_direction": right,
                      "advantage_weighted_dlogp": w,
                      "mean_dlogp_success": sum(d for k, d in delta.items() if adv[k] > 0)
                      / max(1, sum(adv[k] > 0 for k in delta)),
                      "mean_dlogp_failure": sum(d for k, d in delta.items() if adv[k] < 0)
                      / max(1, sum(adv[k] < 0 for k in delta))}))
    return 0


def gate(run_name: str) -> int:
    """Reward-goes-up check over a fixed-slice run's metrics: train reward goes up if the OLS slope of mean reward over >= 10 consecutive
    steps is positive with its 95% interval above zero, OR the last step's mean exceeds step
    0's by more than 2 standard errors (binomial SE of each step's mean, combined)."""
    rows = [json.loads(l) for l in (HERE / "runs" / run_name / "metrics.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if r.get("mean_reward") is not None]
    if len(rows) < 2:
        print("not enough steps")
        return 1
    xs = [float(r["step"]) for r in rows]
    ys = [float(r["mean_reward"]) for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    se_slope = math.sqrt(sum(e * e for e in resid) / max(1, n - 2) / sxx) if n > 2 else float("inf")
    slope_ok = n >= 10 and slope - 1.96 * se_slope > 0

    def se(r):
        k = max(1, r.get("episodes_trained") or r.get("episodes") or 1)
        p = r["mean_reward"]
        return math.sqrt(max(p * (1 - p), 1e-6) / k)
    first, last = rows[0], rows[-1]
    diff = last["mean_reward"] - first["mean_reward"]
    se_diff = math.sqrt(se(first) ** 2 + se(last) ** 2)
    jump_ok = diff > 2 * se_diff
    out = {"steps": n, "slope_per_step": slope, "slope_ci95": [slope - 1.96 * se_slope,
                                                                slope + 1.96 * se_slope],
           "first": first["mean_reward"], "last": last["mean_reward"], "diff": diff,
           "diff_2se": 2 * se_diff, "slope_rule": slope_ok, "jump_rule": jump_ok,
           "PASS": slope_ok or jump_ok}
    print(json.dumps(out, indent=1))
    return 0 if out["PASS"] else 1


def init_adapter(cfg: Config, out: Path, random_b: bool, device: str = "cpu") -> int:
    model = load_policy(cfg, device, random_b=random_b)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    export_for_vllm(out, out / "serve")
    print(f"wrote {out} (+ serve/)  params {lora_param_count(model):,}  "
          f"targets {adapter_targets(model)}")
    return 0


# ------------------------------------------------------------------------------- selftest


def _tiny_model(seed: int = 0):
    """A 2-layer Qwen3-style decoder, random init, no download."""
    from transformers import AutoModelForCausalLM, Qwen3Config  # noqa: PLC0415

    torch.manual_seed(seed)
    c = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                    max_position_embeddings=256)
    return AutoModelForCausalLM.from_config(c).float()


def _selftest() -> int:
    import tempfile

    from peft import LoraConfig, get_peft_model  # noqa: PLC0415

    try:
        from records import AssistantTurn, EnvTurn
    except ImportError:
        from .records import AssistantTurn, EnvTurn  # type: ignore

    failures: list[str] = []

    def check(c: bool, m: str) -> None:
        if not c:
            failures.append(m)

    def ep(eid, gid, reward, prompt_len=6, out=(10, 11, 12), term="natural"):
        p0 = list(range(1, prompt_len + 1))
        o0 = list(out)
        p1 = p0 + o0 + [40, 41]
        o1 = [13, 14]
        return Episode(episode_id=eid, batch_id=0, policy_version="v", task_id="t",
                       group_id=gid, trial_index=0, seed=0,
                       turns=[AssistantTurn(p0, o0, [-1.0] * len(o0), "a", "stop"),
                              EnvTurn("user", "u"),
                              AssistantTurn(p1, o1, [-1.0] * len(o1), "b", "stop")],
                       reward=reward, termination=term, db_hash_before="x", db_hash_after="y",
                       wall_clock_s=1.0, recorded_at="t")

    # --- advantages: 7-of-8 group gives the failing rollout -sqrt(7) (the pass^4 signal)
    g = [ep(f"a{i}", "g1", 1 if i < 7 else 0) for i in range(8)]
    g += [ep(f"b{i}", "g2", 1) for i in range(8)]                 # degenerate all-success
    g += [ep("c0", "g3", 1), ep("c1", "g3", 1, term="length")]   # length trains at reward 0
    g += [ep("d0", "g4", 1), ep("d1", "g4", 1, term="customer_api_error")]  # excluded -> 1 member
    adv, st = group_advantages(g)
    check(abs(adv["a7"] + math.sqrt(7)) < 1e-3, f"near-miss advantage {adv.get('a7')}")
    check(abs(sum(adv[f"a{i}"] for i in range(8))) < 1e-6, "advantages do not sum to 0")
    check(st["degenerate"] == 2 and st["degenerate_all_success"] == 1, f"stats {st}")
    check("b0" not in adv and "d0" not in adv, "degenerate groups got advantage")
    check(adv.get("c1", 0) < -0.99 and adv.get("c0", 0) > 0.99, f"truncated not scored 0 {adv.get('c1')}")

    # --- pass^k: C(c,k)/C(n,k) averaged over tasks
    check(pass_k([(4, 4), (2, 4), (0, 4)], 1) == (1 + 0.5 + 0) / 3, "pass^1 wrong")
    check(abs(pass_k([(4, 4), (2, 4)], 2) - (1 + 1 / 6) / 2) < 1e-9, "pass^2 wrong")
    check(pass_k([(4, 4), (3, 4)], 4) == 0.5 and pass_k([(1, 1)], 2) is None, "pass^4 wrong")

    # --- sequences: mask recovers exactly the generated tokens, positions are shifted by 1
    seqs, sst = build_sequences([g[0]], {"a0": 1.0}, 1000)
    check(len(seqs) == 1 and sst["single_sequence_rate"] == 1.0, f"assembly {sst}")
    s = seqs[0]
    check(s.targets.tolist() == [10, 11, 12, 13, 14], f"targets {s.targets.tolist()}")
    check(all(int(s.token_ids[p + 1]) == int(t) for p, t in zip(s.positions, s.targets)),
          "positions are not target-1")

    # --- chunked head == full log_softmax, and gradients flow only into LoRA
    base = _tiny_model()
    lcfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"],
                      lora_dropout=0.0)
    model = get_peft_model(base, lcfg)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    lp, _ = token_logprobs(model, s, chunk=2, grad=False)
    with torch.no_grad():
        full = torch.log_softmax(model(input_ids=s.token_ids[None]).logits[0].float(), -1)
        ref = full[s.positions, s.targets]
    check(torch.allclose(lp, ref, atol=1e-5), f"chunked logprobs differ {(lp-ref).abs().max()}")

    # --- loss: token-level normalization and TIS truncation
    cfg = Config(lr=1e-2, warmup_steps=1, kl_every=1, head_chunk=2, grad_checkpointing=False)
    s_on = TrainSeq(s.token_ids, s.positions, s.targets, ref.clone(), 1.0, True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    out = train_step(model, opt, [s_on], cfg, 0)
    check(abs(out["loss"] + float(ref.mean())) < 1e-4, f"on-policy loss {out['loss']} vs {-float(ref.mean())}")
    check(out["tis_truncated_frac"] == 0.0 and out["lp_gap_median"] < 1e-5, f"parity {out}")
    check(out["kl_to_base"] is not None and out["kl_to_base"] >= 0, "kl not logged")
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    check(not frozen, f"frozen params received grads: {frozen[:3]}")
    s_off = TrainSeq(s.token_ids, s.positions, s.targets, ref.clone() - 5.0, 1.0, True)
    out2 = train_step(model, opt, [s_off], cfg, 1)
    check(out2["tis_truncated_frac"] == 1.0, f"TIS cap not applied: {out2['tis_truncated_frac']}")

    # --- a positive-advantage step raises the logprob of the trained tokens
    before, _ = token_logprobs(model, s, 2, grad=False)
    s_now = TrainSeq(s.token_ids, s.positions, s.targets, before.clone(), 1.0, True)
    train_step(model, opt, [s_now], cfg, 2)
    after, _ = token_logprobs(model, s, 2, grad=False)
    check(float(after.mean()) > float(before.mean()), "positive advantage did not raise logp")

    # --- sampler: value weighting prefers p~0.85 over p~0.5, and state round-trips
    sm = Sampler(Config(sampler="value", G=8, episodes_per_step=8, seed=1),
                 ["hi", "mid"], {"hi": {"successes": 85, "trials": 100},
                                 "mid": {"successes": 50, "trials": 100}})
    check(sm.weight("hi", 8) > sm.weight("mid", 8) * 3, "value curve shape wrong")
    # ...a U-shaped population (the Sept 25 profile's shape) fits a U-shaped prior, and tasks
    # the profile saw all-success or all-fail are not sampled at all
    pop = {f"t{i}": {"successes": s, "trials": 8}
           for i, s in enumerate([8] * 40 + [0] * 15 + [4] * 20 + [6] * 15 + [2] * 10)}
    sm3 = Sampler(Config(sampler="value", G=8, episodes_per_step=64, seed=2), list(pop), pop)
    check(sm3.a0 < 1 and sm3.b0 < 1, f"prior not U-shaped: Beta({sm3.a0:.2f}, {sm3.b0:.2f})")
    # low-p tasks carry little pass^4 weight by design and are drawn rarely, but the floor
    # means none is starved over a run's worth of steps
    drawn = {t for _ in range(100) for t in sm3.next()[0]}
    check(all(0 < pop[t]["successes"] < 8 for t in drawn), "sampled an all-success/all-fail task")
    check(len(drawn) == 45, f"mixed tasks starved: {len(drawn)} of 45 drawn")
    # ...and the "mixed" weighting prefers the p~0.5 task instead
    smm = Sampler(Config(sampler="value", sampler_weight="mixed", G=8), ["hi", "mid"],
                  {"hi": {"successes": 85, "trials": 100}, "mid": {"successes": 50, "trials": 100}})
    check(smm.weight("mid", 8) > smm.weight("hi", 8), "mixed weighting shape wrong")
    picks = [sm.next()[0][0] for _ in range(200)]
    check(picks.count("hi") > picks.count("mid") * 2, f"sampler picks {picks.count('hi')}")
    st1 = sm.state()
    nxt = sm.next()
    sm2 = Sampler(sm.cfg, ["hi", "mid"])
    sm2.load(json.loads(json.dumps(st1)))
    check(sm2.next() == nxt, "sampler state did not round-trip")
    # ...and a resume that widens the pool keeps the saved posteriors and adds the new tasks
    sm4 = Sampler(Config(sampler="value", sampler_pool="all", G=8, episodes_per_step=64, seed=2),
                  list(pop), pop)
    sm4.load(json.loads(json.dumps(sm3.state())))
    sm5 = Sampler(Config(sampler="value", sampler_pool="not_always", G=8), list(pop), pop)
    check(len(sm5.pool) == 60 and all(pop[t]["successes"] < 8 for t in sm5.pool),
          f"not_always pool wrong: {len(sm5.pool)}")
    check(len(sm4.pool) == 100 and sm4.ab.keys() >= sm3.ab.keys()
          and all(sm4.ab[t] == sm3.ab[t] for t in sm3.ab), "pool widening on resume broke state")
    # --- p4-e: targeted families start at level 1, promote on success, steer by failure class
    check(family_of("tt_items_addr_L2_007") == ("items_addr", 2) and family_of("tb_0015") == (None, 0),
          "family_of wrong")
    fam = ["tt_items_addr_L1_000", "tt_items_addr_L2_000", "tt_pairing_L1_000", "tb_0001"]
    smf = Sampler(Config(sampler="value", sampler_pool="all", G=8, family_adapt=True, retire_n=16),
                  fam)
    check(set(smf.eligible()) == {"tt_items_addr_L1_000", "tt_pairing_L1_000", "tb_0001"},
          f"level gating wrong: {smf.eligible()}")
    E = lambda tid, r, i: Episode(episode_id=f"{tid}-{i}", batch_id=0, policy_version="v",  # noqa: E731
                                  task_id=tid, group_id=tid, trial_index=i, seed=i, turns=[],
                                  reward=r, termination="natural", db_hash_before="", db_hash_after="",
                                  wall_clock_s=0.0, recorded_at="")
    eps_f = [E("tt_items_addr_L1_000", 1, i) for i in range(16)] + [E("tt_pairing_L1_000", 0, i) for i in range(8)]
    ch = smf.update(eps_f, {e.episode_id: ("missed_write" if not e.reward else None) for e in eps_f})
    check(smf.level["items_addr"] == 2 and ch.get("promoted") == ["items_addr->L2"], f"promotion {ch}")
    check("tt_items_addr_L2_000" in smf.eligible(), "promoted level not eligible")
    check(smf.family_mult("tt_items_addr_L2_000") > smf.family_mult("tt_items_addr_L1_000")
          and smf.family_mult("tt_items_addr_L2_000") > smf.family_mult("tt_pairing_L1_000"),
          "failure-class steering wrong")
    check("tt_items_addr_L1_000" in smf.retired, "a 16/16 task was not retired")
    smg = Sampler(smf.cfg, fam)
    smg.load(json.loads(json.dumps(smf.state())))
    check(smg.level == smf.level and smg.retired == smf.retired, "family state did not round-trip")
    # --- p4-e: gated partial credit
    cpc = Config(partial_credit=0.4, unrequested_penalty=0.5)
    check(training_reward(E("t", 1, 0), [], 0, cpc) == 1.0, "success not 1")
    check(training_reward(E("t", 0, 0), [{"name": "cancel_pending_order",
                                          "arguments": {"order_id": "#W1"}}], 0, cpc) == 0.0,
          "no writes should earn nothing")
    check(training_reward(E("t", 0, 0), [], 1, cpc) == -0.5, "unrequested penalty missing")
    cmp_ = Config(partial_credit=0.3, mistake_penalty=0.1, mistake_cap=0.2, credit_mode="graded")
    check(training_reward(E("t", 1, 0), [], 0, cmp_) == 1.0, "a clean success must stay 1")
    check(training_reward(E("t", 0, 0), [{"name": "cancel_pending_order",
                                           "arguments": {"order_id": "#W1"}}], 0, cmp_) == 0.0,
          "graded credit for a failure with no writes must be 0")
    # --- Sept 29: repeated-call penalty, on successes and failures, capped
    import dataclasses  # noqa: PLC0415
    from records import AssistantTurn  # noqa: PLC0415
    rd = lambda n: AssistantTurn([1], [2], [-0.1], "", "tool_calls", tool_calls=[  # noqa: E731
        {"name": "get_order_details", "arguments": {"order_id": "#W1"}}] * n)
    crp = Config(repeat_penalty=0.05, repeat_cap=0.2, mistake_penalty=0.1, mistake_cap=0.2)
    check(training_reward(E("t", 1, 0), [], 0, crp) == 1.0, "no repeats must not cost")
    ok3 = dataclasses.replace(E("t", 1, 0), turns=[rd(3)])
    check(abs(training_reward(ok3, [], 0, crp) - 0.9) < 1e-9, f"2 repeats {training_reward(ok3, [], 0, crp)}")
    ok9 = dataclasses.replace(E("t", 1, 0), turns=[rd(9)])
    check(abs(training_reward(ok9, [], 0, crp) - 0.8) < 1e-9, "repeat cap not applied")
    bad3 = dataclasses.replace(E("t", 0, 0), turns=[rd(3)])
    check(abs(training_reward(bad3, [], 0, crp) + 0.1) < 1e-9, "failures must pay for repeats too")

    # --- checkpoint round-trip gives a bit-exact continuation
    tmp = Path(tempfile.mkdtemp(prefix="trainer-selftest-"))
    try:
        def fresh():
            m = get_peft_model(_tiny_model(), LoraConfig(r=4, lora_alpha=8,
                               target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
            o = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
            return m, o
        cfg2 = Config(lr=1e-2, warmup_steps=1, kl_every=0, head_chunk=4, grad_checkpointing=False)
        mA, oA = fresh()
        for k in range(3):
            train_step(mA, oA, [s], cfg2, k)
        ref_lp, _ = token_logprobs(mA, s, 4, grad=False)
        mK, oK = fresh()
        outK = train_step(mK, oK, [s], Config(lr=1e-2, warmup_steps=1, kl_coef=0.1, head_chunk=4,
                                              grad_checkpointing=False), 0)
        outK2 = train_step(mK, oK, [s], Config(lr=1e-2, warmup_steps=1, kl_coef=0.1, head_chunk=4,
                                               grad_checkpointing=False), 1)
        check(outK2["kl_to_base"] is not None and outK2["kl_to_base"] > 0
              and math.isfinite(outK2["loss"]), f"kl penalty step wrong: {outK2}")

        mB, oB = fresh()
        train_step(mB, oB, [s], cfg2, 0)
        mB.save_pretrained(str(tmp / "ad"))
        save_checkpoint(tmp, {"optimizer": oB.state_dict(), "rng": rng_state()})
        del mB, oB  # "preempted"
        from peft import PeftModel  # noqa: PLC0415
        mC = PeftModel.from_pretrained(_tiny_model(), str(tmp / "ad"), is_trainable=True)
        oC = torch.optim.AdamW([p for p in mC.parameters() if p.requires_grad], lr=1e-2)
        stt = load_checkpoint(tmp)
        oC.load_state_dict(stt["optimizer"])
        set_rng_state(stt["rng"])
        for k in (1, 2):
            train_step(mC, oC, [s], cfg2, k)
        got, _ = token_logprobs(mC, s, 4, grad=False)
        check(torch.allclose(got, ref_lp, atol=1e-6),
              f"resumed run diverged: {(got-ref_lp).abs().max()}")

        # Sept 30: KL anchored to a frozen copy of the starting adapter
        mR = PeftModel.from_pretrained(_tiny_model(), str(tmp / "ad"), is_trainable=True)
        mR.load_adapter(str(tmp / "ad"), adapter_name="ref", is_trainable=False)
        mR.set_adapter("default")
        oR = torch.optim.AdamW([p for p in mR.parameters() if p.requires_grad], lr=1e-2)
        n_def = sum(p.numel() for n, p in mR.named_parameters() if "lora_" in n and ".default." in n)
        check(sum(p.numel() for g in oR.param_groups for p in g["params"]) == n_def,
              "the frozen reference leaked into the optimizer")
        ref_w = {n: p.detach().clone() for n, p in mR.named_parameters() if ".ref." in n}
        cfgR = Config(lr=1e-2, warmup_steps=1, kl_coef=0.1, kl_ref="init", kl_every=1, head_chunk=4,
                      grad_checkpointing=False)
        r0 = train_step(mR, oR, [s], cfgR, 0)
        check(r0["kl_to_ref"] is not None and r0["kl_to_ref"] < 1e-6, f"KL to an identical ref {r0}")
        check(r0["kl_to_base"] is not None and r0["kl_to_base"] > 0, f"base KL not monitored {r0}")
        r1 = train_step(mR, oR, [s], cfgR, 1)
        check(r1["kl_to_ref"] > 0 and math.isfinite(r1["loss"]), f"KL to ref after a step {r1}")
        check(mR.active_adapter == "default", f"active adapter left at {mR.active_adapter}")
        check(all(torch.equal(p, ref_w[n]) for n, p in mR.named_parameters() if ".ref." in n),
              "the reference adapter moved")
        mR.save_pretrained(str(tmp / "adR"), selected_adapters=["default"])
        check(not (tmp / "adR" / "ref").exists(), "the reference was saved with the policy")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("SELFTEST FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("selftest ok")
    return 0


# ------------------------------------------------------------------------------------ CLI


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="trainer.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("probe")
    p.add_argument("--seq-len", type=int, default=12288)
    p = sub.add_parser("parity")
    p.add_argument("records")
    p.add_argument("--limit", type=int, default=16)
    p = sub.add_parser("gate")
    p.add_argument("run")
    p = sub.add_parser("direction")
    p.add_argument("run")
    p.add_argument("records")
    p = sub.add_parser("init-adapter")
    p.add_argument("out")
    p.add_argument("--random", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--run", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--config", type=Path, default=None, help="JSON overrides for Config")
    a = ap.parse_args(argv[1:])

    cfg = Config()
    if getattr(a, "config", None):
        for k, v in json.loads(a.config.read_text()).items():
            if not hasattr(cfg, k):
                raise SystemExit(f"unknown config key {k}")
            setattr(cfg, k, tuple(v) if k == "betas" else v)
    if a.cmd == "selftest":
        return _selftest()
    if a.cmd == "probe":
        return probe(cfg, a.seq_len)
    if a.cmd == "parity":
        return parity(cfg, Path(a.records), a.limit)
    if a.cmd == "gate":
        return gate(a.run)
    if a.cmd == "direction":
        return direction(cfg, a.run, Path(a.records))
    if a.cmd == "init-adapter":
        return init_adapter(cfg, Path(a.out), a.random)
    if a.cmd == "run":
        return run(cfg, a.run, a.resume)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
