# HeuristicRL vs. the colleague's RL-AHD-Crafter

Both projects ask the same question: how should a hand-written heuristic policy H be fed into PPO so that the
result is better or faster than plain PPO, and better than H? They are **not** the same experiment with different
methods: the environment, the kind of heuristic, the regime, the network, the PPO details and the scoring all
differ, and several of those matter more than the methods. Source: the shared `RL-AHD-Crafter-share.zip`
(`train.py`, `heuristic.py`, `score.py`, `run_experiments.sh`, `README.md`); its numbers are the colleague's own runs.

## What is the same

- The question, and the framing of the answer: credit RL by comparing with H, credit H by comparing the steps
  needed to reach a level with plain PPO (our Q3 and Q2).
- H is queried at every training step, alongside the environment.
- The PPO core: clipped objective 0.2, entropy 0.01, value coefficient 0.5, GAE λ 0.95, Adam 3e-4, gradient norm 0.5.
- Two of his three guidance forms exist here too:

| His arm | Ours | Same in substance |
|---|---|---|
| `imitate`, c = 1 annealed to 0 by 25% | Kickstarting (c = 1, to 0 by 50%) | cross-entropy to a_H on the states PPO visits, from scratch |
| `imitate`, c = 0.1 constant | BC → PPO + heuristic anchor (0.1 constant) | the same loss, but ours starts from a BC (or IQL + HUBL) warm start |
| `residual`, β = 2 / 4 / 8 | Residual policy on heuristic (κ = 3) | logits = network + constant · onehot(a_H); PPO on the combined policy; H is part of the deployed policy |
| plain PPO | PPO from scratch | — |

## What differs

### The problem

| | Crafter (his) | Ours |
|---|---|---|
| Problems | 1 | 4 (6 guide experiments) |
| Observation | 64 × 64 pixels, local view + inventory; partially observable | small vector, fully observable |
| Actions | 17, none masked | 2–16, infeasible ones masked |
| Reward | +1 per new achievement (sparse), + 0.1 × Δhealth | dense per step, sums to the metric |
| Episode | until death or 10,000 steps; new random map each time | fixed 100–200 steps |
| Metric | Crafter score: geometric mean over 22 achievement success rates | waste, optimality gap, sojourn time, cost |

### The heuristic

| | His H | Our guides |
|---|---|---|
| Form | 220-line priority program (fight > sleep > drink/eat > tech tree with BFS > plant > explore) | one-line rules (Best Fit, base stock, ...), one LLM-evolved score function |
| Function of what the network sees? | **No**: a hidden random exploration direction, and it reads the decoded semantic map while the network sees pixels | **Yes**: a deterministic function of the observation, so cloning is almost exact |
| Consequence | only "goal steps" (H not exploring, g_t = 1) are guided | every step is guided |
| Known bug | tries to place a furnace with 1 stone (needs 4), kept on purpose | — |

### The regime, the most important difference

| | Crafter | Ours |
|---|---|---|
| H vs plain PPO within budget | H 25.95 vs PPO 7.76 at 1M steps; PPO reaches H only at ~4.4M | PPO from scratch passes the guide in 5 of 6 experiments |
| What the heuristic can buy | a level PPO does not reach in budget | mostly speed and consistency across seeds |

This is why the conclusions point in opposite directions. There, a lasting pull towards H (a strong residual prior)
is the best method; here, every method that holds the policy at H for the whole run ends at or near H.
RL-AHD-OBP (PPO far below an evolved bin-packing teacher) sides with Crafter: a constant anchor won there.

### The guidance details

| | His | Ours |
|---|---|---|
| Warm start before RL | none in any arm | BC, DAgger, IQL, IQL + HUBL, HuRL's value, rollout-improved labels |
| Residual: does the network see a_H? | no; the prior is only added to the logits | yes, onehot(a_H) is also an input |
| Residual prior strength | β = 2, 4, 8 → P(a_H) = 0.32, 0.77, 0.995 at start (17 actions); best β = 4–8 | κ = 3 → about 0.87 (3–4 feasible bin-packing actions), 0.95 (knapsack) |
| Residual on which steps | goal steps only (optionally tech-tree steps only) | all steps |
| Critic warm-up after a warm start | none | 10 critic-only updates for warm starts, the residual policy and the anchor |
| Other ways of using H | — | HuRL, HEPO, Jump-Start RL, IQL / HUBL, DQfD, online IQL, Rollout-DAgger |

### PPO and network

| | His | Ours |
|---|---|---|
| Network | Nature CNN, shared trunk for actor and critic | separate 2 × 128 tanh MLPs |
| Batch per update | 16 envs × 256 steps, 4 epochs × 8 minibatches | 16 envs × 128 steps, 4 epochs, minibatch 512 |
| Discount | γ = 0.99 | γ = 1 (finite horizon) |
| Learning rate | annealed linearly to 0 | constant |

### Evaluation

| | His | Ours |
|---|---|---|
| Score | official Crafter score over **all training episodes**, under the sampling policy (also the last 10%) | greedy policy on 256 fixed held-out episodes at the end; curves on 64 |
| Consequence | a method that starts at H's level is credited for its whole run: "faster" and "better" are mixed | final level and speed are measured separately (Q1–Q3) |
| Seeds | 3; Crafter is not reproducible for a fixed seed | 10; reproducible run by run on one machine |
| Statistics | means and per-seed values (seed spread is very small) | one-sided Welch t-tests, medians for steps |
| Budget | 1M and 5M steps | 1.5M steps |

## Results side by side

| | Crafter (his, 1M / 5M steps) | Ours (6 experiments, 1.5M steps) |
|---|---|---|
| Best method | residual β = 8: 27.97 / 34.39 (H 25.95, plain PPO 7.76 / 19.16) | Rollout-DAgger → PPO + KL; without a simulator IQL (+ HUBL) → PPO + KL |
| Residual | best | κ = 3: no better than PPO from scratch |
| Imitation, annealed | 11.27, better than plain PPO | Kickstarting: good on bin packing, within noise or slightly worse elsewhere |
| Imitation, constant 0.1 | 7.90, the same as plain PPO | constant anchor (after a warm start): among the worst; ends near the guide |

## Cross-test A: his methods on our six experiments

His arms, as in `train.py`, inside our PPO: imitation from scratch (cross-entropy to a_H on every visited state,
coefficient 1 annealed to 0 by 25%, or 0.1 constant), and his residual form (the prior β · onehot(a_H) is only added
to the logits, the networks do not see a_H, the actor starts small, no critic warm-up; the prior is on every step, since
our heuristics have no exploration phase). 10 seeds, 1.5M steps; final metric, lower is better; ▲ / ▼ significantly
better / worse than PPO from scratch (one-sided Welch t-test, 5%).

| Method | BP Best Fit | BP Sum of Sq. | BP EoH | Knapsack | Load bal. | Inventory | better / worse |
|---|---|---|---|---|---|---|---|
| *guide* | 9.68 | 2.36 | 11.48 | 3.69 | 1.73 | 6.94 | |
| PPO from scratch | 3.59 ± 2.14 | 3.59 ± 2.14 | 3.59 ± 2.14 | 1.67 ± 0.25 | 1.64 ± 0.02 | 6.71 ± 0.03 | 0 / 0 |
| **Imitation, annealed by 25% (his)** | 2.11 ± 0.86 ▲ | 1.86 ± 0.10 ▲ | 1.48 ± 0.30 ▲ | 1.66 ± 0.23 | 1.68 ± 0.03 ▼ | 6.68 ± 0.02 ▲ | **4 / 1** |
| Imitation, constant 0.1 (his) | 6.49 ± 2.63 ▼ | 2.14 ± 0.16 ▲ | 3.88 ± 0.50 | 3.08 ± 0.11 ▼ | 1.72 ± 0.01 ▼ | 6.91 ± 0.01 ▼ | 1 / 4 |
| Residual prior β = 2 (his) | 2.84 ± 1.34 | 2.10 ± 0.39 ▲ | 2.40 ± 1.05 | 1.52 ± 0.08 | 1.63 ± 0.01 | 6.76 ± 0.07 ▼ | 1 / 1 |
| Residual prior β = 4 (his) | 2.79 ± 0.82 | 2.23 ± 0.23 ▲ | 2.87 ± 1.30 | 1.44 ± 0.07 ▲ | 1.68 ± 0.04 ▼ | 6.75 ± 0.03 ▼ | 2 / 2 |
| Residual prior β = 8 (his best on Crafter) | 5.05 ± 2.76 | 2.69 ± 0.27 | 3.84 ± 1.22 | 2.34 ± 0.69 ▼ | 1.69 ± 0.04 ▼ | 6.83 ± 0.06 ▼ | 0 / 3 |
| Kickstarting (ours: annealed by 50%) | 2.80 ± 2.39 | 1.43 ± 0.27 ▲ | 1.44 ± 0.30 ▲ | 1.78 ± 0.07 | 1.69 ± 0.02 ▼ | 6.73 ± 0.03 | 2 / 1 |
| Residual policy (ours: κ = 3, network sees a_H) | 3.74 ± 2.40 | 2.55 ± 0.41 | 3.38 ± 2.24 | 2.38 ± 0.91 ▼ | 1.65 ± 0.03 | 6.73 ± 0.04 | 0 / 1 |
| BC → PPO + heuristic anchor (ours, constant 0.1) | 7.35 ± 2.02 ▼ | 2.22 ± 0.21 ▲ | 4.25 ± 0.48 | 3.12 ± 0.17 ▼ | 1.72 ± 0.00 ▼ | 6.90 ± 0.01 ▼ | 1 / 4 |
| Rollout-DAgger → PPO + KL (ours, best) | 1.96 ± 0.23 ▲ | 1.59 ± 0.20 ▲ | 1.70 ± 0.34 ▲ | 1.48 ± 0.24 | 1.60 ± 0.01 ▲ | 6.69 ± 0.03 | 4 / 0 |
| IQL + HUBL → PPO + KL (ours) | 2.09 ± 0.27 ▲ | 1.51 ± 0.18 ▲ | 1.61 ± 0.26 ▲ | 1.55 ± 0.18 | 1.67 ± 0.02 ▼ | 6.72 ± 0.02 | 3 / 1 |

- **His annealed imitation transfers best**: significantly better than PPO from scratch in four experiments, as
  often as Rollout-DAgger, without a warm start or a simulator. Annealing faster (25%) beats our kickstarting
  (50%), and it is the best non-simulator method on inventory (6.68).
- **His constant imitation is as bad here as our constant anchor** (1 / 4), and for the same reason: PPO alone
  passes the guide, so a lasting pull holds it back. On Crafter it was harmless (7.90 vs 7.76).
- **The residual ranking reverses**: the stronger the prior, the worse here (β = 2 best, β = 8 worst, 0 / 3), the
  opposite of Crafter, where β = 8 was best. A strong prior is a lasting pull.

## Cross-test B: our methods on Crafter

`../RL-AHD-Crafter/train_ours.py` (run with `run_cross.sh`, scored with `plot_cross.py`): his plain PPO, annealed
imitation and residual β = 4 / 8 (his `train.py`, unchanged) against our BC / DAgger / IQL / IQL + HUBL → PPO + KL,
HuRL, BC → HuRL and Jump-Start RL. 3 seeds × 1M steps. Our demonstration budget (16,384 steps of H, about 1.6%)
counts against the 1M steps; H's demonstration episodes are not in the Crafter score. BC and DAgger learn only from
H's goal steps. Rollout-DAgger is left out: copying the environment reveals the whole pre-generated map, which the
agent never sees.

| Arm | Crafter score, all episodes | last 10% of steps |
|---|---|---|
| *H alone* | 25.95 | |
| plain PPO (his) | 7.53 (7.9 / 7.3 / 7.4) | 10.22 |
| imitation, annealed by 25% (his) | 10.63 | 14.07 |
| residual β = 4 (his) | 28.33 | 29.76 |
| **residual β = 8 (his)** | **28.71** (28.6 / 28.6 / 28.9) | **30.88** |
| BC → PPO + KL | 9.05 | 11.05 |
| DAgger → PPO + KL | 7.54 | 9.43 |
| **IQL → PPO + KL** (best of ours) | **13.52** (13.5 / 13.2 / 13.8) | **16.75** |
| IQL + HUBL → PPO + KL | 12.48 | 15.54 |
| HuRL | 7.50 | 9.21 |
| BC → HuRL | 8.17 | 9.73 |
| Jump-Start RL | 15.22 | 5.22 |

His arms reproduce his numbers (plain 7.76, imitation 11.27, residual β = 8 27.97). Learning curves:
`../RL-AHD-Crafter/figures/cross.png` (also `results/crafter_cross.png` here).

- **His residual prior wins by far**, above H. On Crafter PPO alone is at a third of H after 1M steps, so a lasting
  pull pays.
- **Of ours, IQL → PPO + KL is best**, ahead of his annealed imitation; IQL + HUBL is close. BC and DAgger → PPO +
  KL are about plain PPO; HuRL does not help.
- **The clones collapse**: the BC and IQL warm starts begin at 14–16 and drop within about 50k steps, despite the KL
  and the critic-only updates (BC to 6, IQL to 10). The DAgger clone starts at only 4.
- **Jump-Start RL** scores high while H plays most of each episode and collapses to 5 when the prefix ends.

## The conclusion both cross-tests support

The ranking of methods is set by the regime, not by the problem's details. A lasting pull towards H (constant
imitation, strong residual prior) is best when PPO alone stays far below H within the budget (Crafter, RL-AHD-OBP)
and costly when PPO alone can pass H (5 of our 6 experiments). On our problems the budget up to which a lasting pull
beats PPO from scratch tracks the steps PPO from scratch needs to pass the guide (`report/regime.py`). Methods whose
pull fades (annealed imitation, KL to a warm start, a performance gate) are the safe choice where PPO alone can pass
H, but on Crafter they reach only about half of the residual prior's score.
