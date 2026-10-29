# Heuristic-guided RL

Eleven ways of using a hand-written heuristic **policy** to help PPO, compared on four toy combinatorial
problems with known heuristics. Full write-up: [`report/report.pdf`](report/report.pdf).

| Problem | Guide heuristic | Reference (stronger) heuristic |
|---|---|---|
| Online bin packing (items 3/4, capacity 12) | Best Fit, Sum of Squares | Sum of Squares |
| Online knapsack | static density threshold | adaptive threshold |
| Load balancing (4 heterogeneous servers) | shortest expected delay | — |
| Lost-sales inventory (lead time 3) | base stock | capped base stock |

Methods: PPO from scratch; BC → PPO (naive, + decaying KL to the clone); DAgger → PPO (naive, + KL);
HuRL (Cheng et al., 2021) from scratch and after BC; HEPO (Lee et al., 2024); IQL → PPO + KL and
IQL + HUBL (Geng et al., 2023) → PPO + KL. The literature methods are adapted to a heuristic policy, as
described in the report.

## Layout

```
hgrl/              problem-agnostic methods: policy, ppo (incl. HuRL shaping), bc (BC, DAgger),
                   guided (HuRL value fit, HEPO, IQL/HUBL), evaluate
problems/          binpack, knapsack, load_balancing, inventory: env + heuristics + plot spec each
run_experiments.py trains all methods (5 seeds) for one problem and guide
plot_results.py    learning curves per experiment + cross-problem overview
results/           <problem>/<guide>/{results.json, summary.md, learning curves}, overview.png, README.md
report/            report.tex / report.pdf; make_tables.py regenerates its tables from results/
tests/             problem contract and method tests
```

## Usage

```bash
pip install -r requirements.txt
python run_experiments.py --problem knapsack                   # the problem's default guide
python run_experiments.py --problem binpack --demo best_fit    # another guide
python run_experiments.py --problem binpack --no-train         # re-summarise / re-plot saved runs
python plot_results.py                                         # all figures + results/overview.png
python report/make_tables.py                                   # tables of the report
python -m pytest tests
```

One experiment (11 methods × 5 seeds × 1.5M steps) takes about 30 min on a 10-core laptop CPU.
