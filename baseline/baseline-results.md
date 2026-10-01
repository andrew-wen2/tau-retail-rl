# Baseline results

*Run Sept 19–20, 2026, on the GCE A100 80GB (`rl-a100-c`, us-central1-c, Spot). Raw results
are in `tau2-bench/data/simulations/baseline_*` on that VM. Numbers below come from
`summarize.py`.*

## Configuration

The official τ³ protocol, except the split (see caveats).

| | |
|---|---|
| Benchmark | τ³-bench v1.0.1, commit `b7ea907`, retail, **`test` split (40 tasks)** |
| Agent | Qwen3.5-2B and -4B under vLLM 0.29.0, text-only, LoRA-free base weights |
| Customer | gpt-5.2, `reasoning_effort: low` |
| Trials | 4, seed 300, `max_steps` 200, `max_concurrency` 16 |
| Sampling, thinking off | temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5 |
| Sampling, thinking on | temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5 |

## Main table (thinking off, the clean runs)

| Model | pass^1 | pass^2 | pass^3 | pass^4 | sec/episode |
|---|---|---|---|---|---|
| **Qwen3.5-4B** | **78.1** | 67.9 | 61.9 | **57.5** | 49.9 |
| **Qwen3.5-2B** | **68.1** | 56.7 | 48.8 | **42.5** | 38.7 |

Leaderboard reference (τ³ v1.0.1, retail, full 114 tasks, Sierra runs):

| Model | pass^1 | pass^4 |
|---|---|---|
| Claude Sonnet 4.5 | 72.4 | 39.5 |
| GPT-5.2 | 81.6 | 51.8 |
| Qwen3.5-397B-A17B (best standard) | 84.4 | 59.6 |

## Full numbers

**Qwen3.5-4B, thinking off** — 160 episodes, no drops, all `user_stop`.
pass^1 78.1 · pass^2 67.9 · pass^3 61.9 · pass^4 57.5 ·
12.6 agent turns · 90,699 prompt tok · 2,450 generated tok · 49.9 s/ep

**Qwen3.5-4B, thinking on** — 18 of 160 episodes (11%) died on infra errors, leaving
1–4 trials per task, so pass^k is not comparable.
pass^1 84.8 excluding the dead episodes, or **76.2 / pass^4 50.0** counting them as failures
(what tau2's own summary reports) ·
11.8 turns · 77,526 prompt tok · 4,530 generated tok · 69.5 s/ep

**Qwen3.5-2B, thinking off** — 160 episodes, 159 `user_stop` and 1 `too_many_errors`.
pass^1 68.1 · pass^2 56.7 · pass^3 48.8 · pass^4 42.5 ·
13.5 turns · 99,031 prompt tok · 2,678 generated tok · 38.7 s/ep

**Qwen3.5-2B, thinking on** — 1 episode (1%) dropped, 3–4 trials per task.
pass^1 68.5 · pass^2 56.2 · pass^3 48.1 ·
13.1 turns · 90,136 prompt tok · 7,130 generated tok · 63.6 s/ep

## Findings

**1. Thinking mode: off.** For the 2B it buys nothing measurable — 68.5 vs 68.1 pass^1, well
inside noise — while costing 1.6× the wall clock and 2.7× the generated tokens. For the 4B it
can't even be scored because of the empty-message failures. Training and all further evals use
thinking off.

**2. Success is not sparse.** I had assumed a cold start (about 5% success). The real base
rates are 68% (2B) and 78% (4B). Consequences:
- A supervised warm start before RL is unnecessary, so none was built.
- Most GRPO groups will be all-success and produce **zero advantage**. The learning edge is the
  ~30% of tasks the model sometimes fails, so **task selection matters more than warm-starting**:
  filter or oversample tasks whose group outcomes are mixed.

**3. Reliability is where the headroom is.** Both models are about 10 points off a
leaderboard-topping pass^1, but the 2B is **17 points** below the best pass^4. With one test
task worth 2.5 points and differences under ~10 points not meaningful, a claim needs a big
move. The 2B's pass^4 is the only gap large enough for an unambiguous claim.

**4. The A100 is not the bottleneck.** During the runs vLLM reported `Running: 1–2 reqs`,
`GPU KV cache usage: 0.8%`, and 87.2% prefix cache hits. The agent generates a turn in a
second or two, then waits on the gpt-5.2 customer. RL throughput will come from running many
episodes concurrently, not from a faster GPU — which also means the A100-vs-H100 choice matters
less than I had assumed.

## Empty assistant messages with thinking on

11% of the 4B's thinking-on episodes died with:

```
AssistantMessage must have either content or tool_calls
```

The model returned a message with no text and no tool call; τ³ retried 4 times, then dropped the
episode. The 2B hit this once (1%). Suspected cause is `--reasoning-parser qwen3` in `serve.sh`
moving everything into `reasoning_content` and leaving `content` empty, but this is unconfirmed.

It does not affect training or the final eval, which run with thinking off. The rollout driver
treats an empty assistant message as a terminal reward-0 episode and counts it.

**Methodological note:** infra errors must be handled consistently. tau2's own summary counts
them as failures in pass^k but excludes them from average reward, which is why the same run
reads 76.2 or 84.8 pass^1 depending on the convention. `summarize.py` excludes them and prints
the drop count, so the exclusion is always visible.

## Measured noise (bootstrap over tasks, 2000 resamples, `ci.py`)

| Run | pass^1 | pass^2 | pass^4 |
|---|---|---|---|
| 2B, think off | 68.1 **[56.9, 78.8]** | 56.7 [43.3, 69.2] | 42.5 **[27.5, 57.5]** |
| 4B, think off | 78.1 **[68.1, 87.5]** | 67.9 [55.0, 80.0] | 57.5 **[42.5, 72.5]** |

**±11 points on pass^1 and ±15 on pass^4**, wider than a ~10-point rule of thumb. The
two models' intervals overlap heavily despite a 10-point gap.

Paired, on the same 40 tasks (`paired.py`, 4B − 2B):

| | difference | 95% CI |
|---|---|---|
| pass^1 | +10.0 | [+0.6, +18.8] |
| pass^2 | +11.2 | [−2.1, +25.0] |
| pass^4 | +15.0 | [−5.0, +32.5] |

Pairing narrows pass^1 from ~31 points wide to 18, but **a 15-point pass^4 gap between a 2B and
a 4B still spans zero**. On this eval set RL would need about +10 pass^1 or +20 pass^4 to be
significant.

**This is why the split changed (Sept 20).** Training moves to the converted τ-bench 500 alone,
and all 114 τ³ retail tasks become the eval set: intervals shrink by ~40%, the detection
threshold drops to roughly +6 pass^1 and +12–15 pass^4, and the numbers become directly
comparable to Sierra's leaderboard, which uses those same 114 tasks. `run_baseline.sh` now defaults to `SPLIT=base`.

- **Every leaderboard comparison is unpaired** — different task set, different run — so it
  inherits these widths. A claim like "beats Sonnet 4.5's 72.4 pass^1" needs a 15–20 point gain
  to be defensible as stated.
- **Pairing is the fix.** Base vs trained on the same 40 tasks cancels the task-difficulty
  variance that drives these widths. Never report an unpaired difference. `paired.py` computes
  the paired bootstrap.
- More trials shrink the trial component but not the task-sampling component, which dominates
  here. n=8 helps pass^4 most, since each task stops being all-or-nothing.
- **Report pass^2 as well as pass^4.** It measures reliability with materially less noise.

## Caveats

- **Split mismatch.** These are the 40-task `test` split; leaderboard numbers are the full 114.
  Not a like-for-like comparison; the 114-task re-baseline below is.
- The 4B thinking-off run was resumed after a Spot preemption. Same config, same seed; τ³ keeps
  completed episodes in `results.json` and reruns only the missing ones.

## Environment fixes needed on the GCE Deep Learning VM image

All now folded into `setup.sh`:
- `python3.12-dev`, `build-essential` — Triton compiles `cuda_utils.c` against Python headers.
- `ninja-build` — FlashInfer JIT-builds its sampling kernel.
- `websockets` — τ³ imports its voice module at startup, but the dependency ships only in the
  optional `[voice]` extra. Installed alone; the full extra needs `pyaudio`.

Confirmed on the good path: `Using Triton/FLA GDN prefill kernel`, `GDN decode kernel: cuda` —
not the slow pure-torch DeltaNet fallback.

## The 114-task re-baseline (Sept 20) — operational findings

*Qwen3.5-2B, all 114 retail tasks, 8 trials, thinking off, seed 300, official protocol,
`b7ea907`, vLLM 0.29.0. Full output in `results/base_gpt52/summary.txt`, raw trajectories in
`results/base_gpt52/results.json`. 912/912 episodes, no drops after the refill; endings were 909
`user_stop`, 2 `too_many_errors`, 1 `max_steps`.*

| | pass^1 | pass^2 | pass^4 | pass^8 |
|---|---|---|---|---|
| **Qwen3.5-2B, 114 tasks, n=8** | **68.2** | 54.5 | **39.8** | 27.2 |
| *(40-task `test`, n=4)* | *68.1* | *56.7* | *42.5* | *-* |

**pass^1 reproduced almost exactly** (68.2 vs 68.1) on nearly three times the tasks, which is
the strongest evidence yet that the 40-task number was not a fluke. pass^4 came in 2.7 points
lower, well inside the interval.

These are now **directly comparable to Sierra's leaderboard**, same 114 tasks and protocol.
The 2B base already sits at Claude Sonnet 4.5's pass^4 (39.8 vs 39.5) while trailing it on
pass^1 (68.2 vs 72.4) - it is nearly as *reliable* as Sonnet 4.5 and less *capable*, which is
exactly the shape RL on reliability is supposed to exploit.

**Headroom.** The band of tasks solved in 4 to 7 of 8 trials is **48 of 114 (42%)**. About 25
such tasks are needed for a pass^4 gain to be within reach, and the 40-task pass^k had
suggested ~29; the real number is two-thirds larger.

| $c/n$ | tasks | |
|---|---|---|
| 0/8 never solved | 8 (7%) | zero gradient |
| 1/8 - 3/8 | 14 (12%) | |
| 4/8 - 7/8 | 61 (54%) | the band |
| 8/8 always solved | 31 (27%) | zero gradient |

**75 of 114 tasks (66%) come out mixed**, which is the GRPO gradient supply and is far
healthier than the 40-task run's 'most groups will be all-success' worry. Only 39 tasks are flat.
Taking the band to $p=0.95$ and changing nothing else gives **pass^4 62.3**, above the best
standard leaderboard model (59.6). That is a plug-in estimate on measured rates and it is
optimistic - it assumes every band task is convertible, when some of that variance belongs to
the customer simulator, not the agent - so treat it as the ceiling, not the forecast.

*The operational findings below are what the scores do not show.*

**1. The judge is a rate-limit bottleneck.** 492 `RateLimitError` hits across the run, all
gpt-4.1 against a **30,000 TPM** org cap. Most were retried successfully; **23 episodes
exhausted their retries and died** with `termination_reason: infrastructure_error`, zero
messages and `duration: 0.0` — they never ran at all. They clustered on 8 tasks (task 107 lost
4 of 8 trials; 103, 104, 29, 111, 37 lost 3 each), which is worse than a flat 2.5% loss because
it leaves those tasks at a different `n`. **Refilled at `CONC=12`** after stripping them, with
the pre-repair file kept as `results.pre_repair.json`. The gpt-4.1 TPM limit was raised afterwards.

**2. Concurrency: 40 works, and 40 is the ceiling.** At `--max-concurrency 16` the first runs measured
the A100 nearly idle (1–2 running requests, 0.8% KV cache) because the agent waits on the
customer. At **40** the GPU ran at 100% and the run finished in ~55 minutes instead of a
projected 37+ at 16 — but 40 is also what pushed the judge over its TPM cap. `CONC` is now a
variable in `run_baseline.sh`, defaulting to 16 so the first runs reproduce.
**`sec/episode` from the re-baseline is not comparable to the numbers above**: it blends two
concurrency regimes and includes rate-limit retry latency.

**3. A second pathological-generation mode.** One episode (task 98, trial 4) generated
**210 tok/s for 35+ minutes with prompt throughput at 0.0** — a single unbounded completion,
not a sequence of turns — and was killed manually. This is distinct from the thinking-on
empty-message bug above, and it happened with `presence_penalty 1.5` already applied.
At 1-in-912 it is a rounding error for eval. **It is not a rounding error for GRPO:** a rollout
that never returns stalls its whole group, and a stalled group contributes no advantage. The
loop needs a per-rollout timeout.

**4. `--auto-resume` is mandatory for unattended runs.** τ³ prompts `results.json already
exists. Do you want to resume the run? (y/n)` before resuming. In a detached tmux session
nothing answers, and the run sits idle with the GPU at 0% looking healthy. Now passed by
`run_baseline.sh`; the `_n8` directory naming is what makes it safe, since a resume can only
land where size, domain, split, mode and trial count already match.

## Infrastructure notes (Sept 20)

- **The A100 in `rl-a100-c` failed.** `NVRM: _kgspBootGspRm: unexpected WPR2 already up`, then
  `Xid 120, GSP task exception` on a cold boot: the GSP firmware processor on the card was
  wedged. Not fixable from the guest — module reload, PCI reset and two stop/starts all failed,
  because a stop/start can return you to the same host. Fixed by snapshotting the boot disk and
  recreating in another zone (`rl-a100-a`, us-central1-a).
- **Spot capacity in us-central1 is tight.** Both zones returned `stockout` repeatedly, and one
  instance was preempted 3.5 minutes into the run. Resume works (τ³ keeps completed episodes),
  so a start-serve-resume loop converges. On-demand quota was **denied**, so Spot is the only
  option until the account has more billing history.
- **A disk snapshot is a point-in-time copy of the scripts too.** The rebuilt VM came up with a
  `run_baseline.sh` that predated the Sept 20 edits — it would have run the 40-task `test` split
  at 4 trials into an old results directory. Re-push the scripts after any rebuild.
