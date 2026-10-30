# Heuristic-guided RL

21 ways of using a heuristic **policy** to help reinforcement learning, plus 19 follow-up variants, compared on four
toy combinatorial problems with known heuristics (six guides, 10 seeds, 1.5M steps each), and cross-tested on a
colleague's Crafter set-up (`docs/comparison_crafter.md`). Full write-up: [`report/report.pdf`](report/report.pdf).

| Problem | Guide heuristic | Reference (stronger) heuristic |
|---|---|---|
| Online bin packing (default; EoH's Weibull items, capacity 100) | EoH-evolved (from RL-AHD-OBP), Best Fit, Sum of Squares | EoH-evolved |
| Online bin packing, toy (`binpack_toy`; items 3/4, capacity 12) | Best Fit, Sum of Squares, EoH-evolved | Sum of Squares |
| Online knapsack | static density threshold | adaptive threshold |
| Load balancing (4 heterogeneous servers) | shortest expected delay | — |
| Lost-sales inventory (lead time 3) | base stock | capped base stock |

Methods:
- **offline warm starts:** BC and DAgger → PPO (naive, + decaying KL), HuRL (from scratch, after BC), IQL and
  IQL + HUBL → PPO + KL;
- **heuristic during RL:** kickstarting, BC / IQL + HUBL → PPO + constant heuristic anchor (also with a 4× batch),
  residual policy on the heuristic, Jump-Start RL, HEPO;
- **model-based teacher:** Rollout-DAgger, the heuristic improved by Monte-Carlo rollouts on a simulator and
  distilled with DAgger → PPO + KL (decaying or held);
- **off-policy with the heuristic's data:** DQfD, IQL offline → online.
- **follow-ups:** the Crafter study's arms (annealed / constant imitation, residual prior β = 2/4/8), fading versions
  of the methods that hold the pull, a Q-filtered and a performance-gated imitation, rollout labels without the
  significance test, online IQL with normalised advantage weights.

The regime decides: a lasting pull towards the heuristic pays only until RL alone would pass the heuristic, so it is
the best choice on Crafter and among the worst here. Most reliable here: Rollout-DAgger → PPO + KL (needs a simulator,
80–500M simulated steps per run); without a simulator, IQL (+ HUBL) → PPO + KL or performance-gated imitation. Online
IQL with normalised advantage weights is the best method on knapsack and inventory; with γ = 0.99
it is second overall (best on Sum of Squares and load balancing) but fails on knapsack.

## Layout

```
hgrl/              problem-agnostic methods: policy, ppo (incl. HuRL shaping, anchor, JSRL), bc (BC, DAgger),
                   guided (HuRL value fit, HEPO, IQL/HUBL), online (heuristic in the observation, residual
                   policy), rollout (rollout improvement), offpolicy (DQfD, online IQL), evaluate
problems/          binpack (Weibull), binpack_toy, knapsack, load_balancing, inventory: env + heuristics + plot spec each
run_experiments.py trains all methods for one problem and guide
rollout_reference.py  each guide and its rollout improvement as policies -> results/rollout_reference.json
plot_results.py    learning curves per experiment + cross-problem overview
results/           <problem>/<guide>/{results.json, summary.md, learning curves}, overview.png, README.md
report/            report.tex / report.pdf; make_tables.py regenerates its tables, regime.py the regime figure
docs/              comparison_crafter.md: the colleague's Crafter study vs. this one, and both cross-tests
tests/             problem contract and method tests
```

## Usage

```bash
pip install -r requirements.txt
python run_experiments.py --problem knapsack                   # the problem's default guide
python run_experiments.py --problem binpack --demo best_fit    # another guide (default: EoH-evolved)
python run_experiments.py --problem binpack_toy                # the small discrete bin packing
python run_experiments.py --problem binpack --no-train         # re-summarise / re-plot saved runs
python plot_results.py                                         # all figures + results/overview.png
python report/make_tables.py                                   # tables of the report
python -m pytest tests
python run_experiments.py --problem knapsack --seeds 10 --reuse --variants dqfd   # one method, keep the rest
python rollout_reference.py                                    # guides vs. their rollout improvement
```

The 21 main methods × 10 seeds for the six experiments took about 1.5 h on a 96-core machine, mostly the
Rollout-DAgger runs (15–70 min each on one core). Results are reproducible run by run only on the same machine.
