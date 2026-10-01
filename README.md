# Multi-turn GRPO for a 2B customer-service agent on τ³-bench retail

**Qwen3.5-2B** (LoRA, thinking off) trained with a multi-turn GRPO loop written from scratch, to
act as the agent in **τ³-bench retail** (Sierra's `tau2-bench` v1.0.1). The agent handles an online
store's customers: it cancels and modifies orders, processes returns and exchanges, and follows a
written policy through tools that read and write the store's database.

**Result: on the 114 held-out τ³ retail tasks, the trained model is indistinguishable from the
untrained one.** Training did produce a gain of about 10 pass^4 points, but only against the
simulated customer used in training. With the benchmark's customer the gain is gone.

| Qwen3.5-2B, τ³ retail, 114 tasks × 8 trials | pass^1 | pass^2 | pass^4 |
|---|---|---|---|
| Base (no training) | 68.2 | 54.5 | 39.8 |
| Trained | 68.5 | 54.3 | 39.1 |
| **Difference, paired, 95% CI** | **+0.3 [−3.6, +4.3]** | **−0.2 [−5.2, +5.0]** | **−0.6 [−7.1, +6.0]** |

pass^k is the probability that k independent attempts at the same task all succeed. Both models
ran under the official protocol (gpt-5.2 customer at low reasoning, seed 300) on the same pinned
commit. Intervals come from a paired bootstrap over tasks. The target set before training was
pass^4 ≥ 52, which would have beaten GPT-5.2's 51.8. It was not reached.

For scale, Sierra's leaderboard on the same tasks and protocol:

| Model | pass^1 | pass^4 |
|---|---|---|
| Claude Sonnet 4.5 | 72.4 | 39.5 |
| **Qwen3.5-2B, base / trained (this repo)** | **68.2 / 68.5** | **39.8 / 39.1** |
| GPT-5.2 | 81.6 | 51.8 |
| Qwen3.5-397B-A17B | 84.4 | 59.6 |

The training tasks derive from τ-bench's retail train set, so a leaderboard entry would count as
"custom", not "standard".

## What training changed

### The gain depends on the customer

Training used DeepSeek-V4-Flash as the simulated customer. The benchmark uses gpt-5.2. Gains over
the base model, on the same tasks:

| Tasks | Customer | pass^1 gain | pass^4 gain |
|---|---|---|---|
| 166 held-out tasks built like the training tasks | DeepSeek | +6.2 [+3.6, +8.9] | **+10.4 [+5.6, +15.2]** |
| τ³ 114 | DeepSeek | +4.4 [−0.5, +9.4] | **+10.8 [+1.8, +20.2]** |
| τ³ 114 | gpt-5.2 | +0.3 [−3.6, +4.3] | −0.6 [−7.1, +6.0] |

The first row is an earlier checkpoint; the other two are the final one. On τ³'s own tasks the gain
survives when the customer is DeepSeek and vanishes when it is gpt-5.2. The difference between
those two gains is +4.1 pass^1 [−1.9, +9.8] and +11.4 pass^4 [+0.6, +22.4], so only the pass^4
difference excludes zero.

The base model scores 62.7 pass^1 with DeepSeek and 68.2 with gpt-5.2. The trained model scores
67.1 and 68.5. Training closed a gap that only the DeepSeek customer opens; gpt-5.2 never exposed
that weakness, so with it there was nothing to gain. Whether the model learned to exploit DeepSeek
or to recover from problems DeepSeek causes is not settled.

The DeepSeek runs on the 114 used 4 trials, and three base tasks have only 3, so their pass^4 is
over 111 tasks.

### It improved what the training tasks emphasized, and learned to act when it should not

An exploratory breakdown of the 114 tasks by type, with the gpt-5.2 customer:

| τ³ task type | Tasks | pass^1 change | Share of training |
|---|---|---|---|
| Changes the address on a pending order | 20 | **+15.6 [+6.9, +25.0]** | heavy |
| Three or more database writes | 21 | +7.1 [−3.0, +17.3] | heavy |
| One write | 60 | −0.8 [−6.5, +4.6] | light |
| **No write is correct** | 10 | **−10.0 [−21.2, +1.2]** | about 1%, against 9% of τ³ |

On tasks where the right move is to answer or refuse, the trained model failed 18 of 80 episodes
against the base model's 10, and in 15 of those 18 it made a write anyway. Training almost never
included a task whose right answer was to do nothing, so it taught a bias toward acting.

### It spirals after a rejected write

When a tool rejects a write, for example a second return on an order whose status the first return
already changed, the trained model deliberates inside its reply until it hits the length limit.
The median reply right after a tool error grew from 226 tokens (base) to 652 over training, while
every other reply stayed near 117. Training reached that state in 2–3% of episodes; the gpt-5.2
customer leads to it in about 20%.

### It did not exploit the reward's blind spot

The reward checks only the final database state, so it cannot see whether the agent authenticated
the customer before writing, which the policy requires. Skipping authentication was a shorter
route to full reward on every training task. The trained model still authenticated before its
first write in 99.8% of final-eval episodes, against 99.9% for the base model.

## What moved the score during training

Measured with the training customer on held-out tasks built like the training tasks.

- **Redesigning the tasks produced the only big jump.** Rewriting the training tasks into τ³'s
  structure gave +5.8 pass^1 and +7.4 pass^4: requests spanning several orders, vague targets,
  changes of mind, refusals, and customer instructions that no longer script the conversation.
  Later changes to the task pool, KL penalty, group size and learning rate all plateaued.
- **Averaging weights beat every single checkpoint.** A uniform average of four runs' LoRA weights
  gave the best result, pass^4 +10.4.
- **Supervised fine-tuning on a stronger model's successes erased most of the gain.** pass^4 fell
  by 8.0 [−12.7, −3.6] even as held-out loss halved.
- **The stable learning-rate band was narrow**, and stable training metrics did not predict
  transfer to the benchmark's customer.
- **Mixing in a second, cheaper customer did not help measurably.** The cheapest candidates looped
  or misstated their own tasks.

## Method

Written from scratch: the rollout driver, the episode record format, the trainer, the task sampler
and checkpointing. Borrowed: vLLM, PEFT and the τ³ environment.

- **Record first, train offline.** `driver.py` runs episodes against a vLLM server and a simulated
  customer, keeping the generated token ids of every turn instead of re-rendering the history.
- **GRPO in `trainer.py`.** Group-normalized advantages on agent tokens only, truncated importance
  sampling against the gap between serving and training logprobs, and a KL penalty. LoRA covers
  attention, the DeltaNet projections and the MLP.
- **Task sampling aimed at reliability.** Each task carries a posterior over its success rate, and
  sampling favors tasks whose groups are likely to mix successes and failures, since uniform
  groups carry no advantage.
- **Guards.** Per-rollout timeout, per-turn token cap, empty replies scored as failures, and
  checkpoint-and-resume that survives a preempted machine.
- **Evaluation.** Base and trained models run the same tasks and are compared with a paired
  bootstrap (`baseline/paired_final.py`, `baseline/paired_customers.py`).

## Limitations

- **The task design was informed by τ³.** Later task generators were written to match τ³'s task
  structure after reading its tasks. No τ³ task, instruction or trajectory was trained on.
- **Three leaking tasks were trained on.** They share a user with held-out tasks and were used in
  two early runs that every later checkpoint descends from.
- **The training reward was a subset of the benchmark's.** τ³ also scores natural-language
  assertions with an LLM judge on 40 tasks; training never did.
- **The checkpoint was chosen on other tasks.** Selection used 94 non-τ³ tasks with a gpt-5.2
  customer. No checkpoint was chosen on the 114.
- **The trained model had a reply cap.** 8,192 tokens per reply; the base model ran uncapped and
  never exceeded 5,967.
- **The customer comparison is thin.** Two customers, 4 trials against 8, and a post hoc task-type
  breakdown.

## Repository layout

| Path | Contents |
|---|---|
| `driver.py`, `records.py`, `trainer.py` | Rollout driver, episode records, GRPO trainer |
| `fidelity.py`, `sft.py` | Check of the driver against τ³'s own runner; the fine-tuning experiment |
| `taskgen/`, `tasks/` | Task conversion and generation scripts; the task sets, splits and reports |
| `data/retail/` | τ³'s retail database and policy at the pinned commit |
| `experiments/` | Per-run configs, weight averaging and offline analyses |
| `baseline/` | Machine setup, serving, run scripts, the eval harness and `baseline-results.md` |
| `results/` | Summaries and paired analyses per model and customer (`soupm` is the final checkpoint, `soup4` the earlier one) |

## Directions that would change the outcome

- **Train against a population of customers.** One simulated customer defines which failures the
  agent ever sees. A population that includes customers as strong as the benchmark's would make
  the gradient reflect customer behavior in general, not one model's habits.
- **Train on the benchmark's full reward.** Score natural-language assertions and required
  disclosures during training, and penalize writes when no write is correct, so the objective is
  the one being evaluated.
- **Assign credit per turn.** One scalar per 26-message episode is a weak signal. Branching several
  continuations from a shared conversation prefix, with the database restored to that point, would
  give advantages at the turn where outcomes diverge.
- **Start rollouts from the states that go wrong.** Rejected writes and mid-conversation changes of
  mind are rare under the training customer and common in evaluation. Seeding episodes from those
  states would train recovery directly.
