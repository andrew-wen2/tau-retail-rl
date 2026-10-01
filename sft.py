"""Supervised fine-tune of the LoRA policy on successful conversations (Sept 28).

GRPO only reinforces what the 2B already does sometimes; on the tasks it rarely or never
solves there is nothing to reinforce. This teaches those behaviors directly from successful
DeepSeek-V4.1-Flash agent transcripts, mixed with the 2B's own successes against forgetting.
The data (sft_v1.json) is built by sft_dump.py / sft_render.py and is already token ids with
a loss mask on agent tokens only, rendered with the served template and checked token-for-token
against recorded rollouts.

Loss is token-level mean NLL over the batch's trained tokens, the same normalization the GRPO
step uses. A held-out slice of the teacher examples (by task) gives an overfitting signal.

    python sft.py <init adapter dir> <run name> [--epochs 2] [--lr 5e-5] [--batch 16]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from trainer import HERE, Config, TrainSeq, load_policy, save_adapter, token_logprobs


def to_seq(ex: dict) -> TrainSeq:
    ids, mask = ex["input_ids"], ex["loss_mask"]
    idx = [i for i, m in enumerate(mask) if m and i > 0]
    return TrainSeq(token_ids=torch.tensor(ids, dtype=torch.long),
                    positions=torch.tensor([i - 1 for i in idx], dtype=torch.long),
                    targets=torch.tensor([ids[i] for i in idx], dtype=torch.long),
                    behavior_lp=torch.zeros(len(idx)), advantage=1.0, single=True)


def nll(model, seqs: list[TrainSeq], cfg: Config, grad: bool) -> float:
    total = sum(len(s.targets) for s in seqs)
    out = 0.0
    for s in seqs:
        lp, _ = token_logprobs(model, s, cfg.head_chunk, grad=grad)
        loss = -lp.sum() / total
        if grad:
            loss.backward()
        out += float(loss.detach())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("init_adapter")
    ap.add_argument("run")
    ap.add_argument("--data", default="sft_v1.json")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--heldout_tasks", type=int, default=15)
    a = ap.parse_args()

    cfg = Config()
    run_dir = HERE / "runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    data = json.load(open(HERE / a.data))
    rng = random.Random(0)
    teacher_tasks = sorted({d["task_id"] for d in data if d["source"] == "teacher"})
    held = set(rng.sample(teacher_tasks, a.heldout_tasks))
    train = [d for d in data if d["task_id"] not in held and len(d["input_ids"]) <= cfg.max_seq_len]
    heldout = [to_seq(d) for d in data if d["task_id"] in held and d["source"] == "teacher"]
    print(f"train {len(train)} ({sum(d['source'] == 'teacher' for d in train)} teacher) | "
          f"held-out teacher examples {len(heldout)} on {len(held)} tasks", flush=True)

    model = load_policy(cfg, "cuda", adapter_dir=Path(a.init_adapter))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=0.0)
    steps_per_epoch = (len(train) + a.batch - 1) // a.batch
    total_steps = steps_per_epoch * a.epochs
    log = open(run_dir / "sft_metrics.jsonl", "a")

    def heldout_loss() -> float:
        model.eval()
        with torch.no_grad():
            return nll(model, heldout, cfg, grad=False)

    h0 = heldout_loss()
    print(f"held-out NLL before: {h0:.4f}", flush=True)
    log.write(json.dumps({"step": 0, "heldout_nll": h0}) + "\n"); log.flush()
    step = 0
    for ep in range(a.epochs):
        order = train[:]
        rng.shuffle(order)
        for b in range(0, len(order), a.batch):
            t0 = time.time()
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = nll(model, [to_seq(d) for d in order[b:b + a.batch]], cfg, grad=True)
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip))
            # linear warmup, then cosine to 10% of the peak
            if step < a.warmup:
                lr = a.lr * (step + 1) / a.warmup
            else:
                prog = (step - a.warmup) / max(1, total_steps - a.warmup)
                lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + __import__("math").cos(__import__("math").pi * prog)))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            step += 1
            rec = {"step": step, "epoch": ep, "train_nll": loss, "grad_norm": gnorm, "lr": lr,
                   "t_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
        h = heldout_loss()
        version, path = save_adapter(model, run_dir, ep + 1)
        rec = {"epoch": ep + 1, "heldout_nll": h, "adapter": version}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    print("SFT-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
