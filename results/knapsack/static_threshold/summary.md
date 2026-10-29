# Online knapsack: guided by Static threshold

Gap to offline optimum (%) (lower is better), greedy policies on 256 fixed episodes
(200 items, weight and value i.i.d. uniform on {1..10}, capacity 250).

| Heuristic | Metric |
|---|---|
| Adaptive threshold | 1.24% ± 0.72 |
| Static threshold (guide) | 3.69% ± 3.04 |
| Greedy | 54.48% ± 4.80 |
| Random | 54.95% ± 4.41 |

| Variant | Initial | Final | Worst during training | Steps to beat Static threshold (median) | Seeds beating Static threshold |
|---|---|---|---|---|---|
| PPO from scratch | 77.67% | 1.45% ± 0.09 | 53.88% | 225 k | 5/5 |
| BC → PPO (naive) | 4.76% | 1.64% ± 0.25 | 7.84% | 136 k | 5/5 |
| BC → PPO + KL (decay 50%) | 4.76% | 1.56% ± 0.06 | 4.87% | 115 k | 5/5 |
| BC → PPO + KL (decay 20%) | 4.76% | 1.70% ± 0.19 | 5.07% | 115 k | 5/5 |
| DAgger → PPO (naive) | 4.56% | 1.82% ± 0.16 | 6.91% | 177 k | 5/5 |
| DAgger → PPO + KL (decay 50%) | 4.56% | 1.77% ± 0.15 | 4.70% | 484 k | 5/5 |
| HuRL (from scratch) | 77.67% | 1.74% ± 0.21 | 55.07% | 115 k | 5/5 |
| BC → HuRL | 4.76% | 1.63% ± 0.14 | 5.94% | 115 k | 5/5 |
| HEPO (from scratch) | 77.67% | 3.05% ± 0.21 | 8.68% | 123 k | 5/5 |
| IQL → PPO + KL | 7.11% | 1.55% ± 0.15 | 7.11% | 115 k | 5/5 |
| IQL + HUBL → PPO + KL | 4.57% | 1.44% ± 0.10 | 4.66% | 13 k | 5/5 |

Variants that use Static threshold offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its 12,800 steps in their step counts; HEPO's heuristic rollouts are half of its PPO steps. 'Beat' is measured on the 64 evaluation episodes of the learning curves.
