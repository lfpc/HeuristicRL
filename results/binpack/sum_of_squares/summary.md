# Online bin packing: guided by Sum of Squares

Wasted bin capacity (%) (lower is better), greedy policies on 256 fixed episodes
(B = 12, item sizes {3, 4}, p = {2/3, 1/3}, 200 items).

| Heuristic | Metric |
|---|---|
| Best Fit | 9.68% ± 0.99 |
| Sum of Squares (guide) | 2.36% ± 0.66 |
| Random | 13.56% ± 2.70 |

| Variant | Initial | Final | Worst during training | Steps to beat Sum of Squares (median) | Seeds beating Sum of Squares |
|---|---|---|---|---|---|
| PPO from scratch | 58.87% | 2.48% ± 0.85 | 14.05% | 1157 k | 4/5 |
| BC → PPO (naive) | 2.49% | 1.95% ± 0.28 | 4.92% | 33 k | 5/5 |
| BC → PPO + KL (decay 50%) | 2.49% | 1.75% ± 0.16 | 2.65% | 54 k | 5/5 |
| BC → PPO + KL (decay 20%) | 2.49% | 1.78% ± 0.08 | 3.45% | 74 k | 5/5 |
| DAgger → PPO (naive) | 2.48% | 1.94% ± 0.14 | 5.20% | 54 k | 5/5 |
| DAgger → PPO + KL (decay 50%) | 2.48% | 1.68% ± 0.09 | 2.58% | 279 k | 5/5 |
| HuRL (from scratch) | 58.87% | 1.99% ± 0.38 | 19.94% | 1221 k | 5/5 |
| BC → HuRL | 2.49% | 2.07% ± 0.13 | 6.47% | 730 k | 5/5 |
| HEPO (from scratch) | 58.87% | 2.40% ± 0.05 | 12.98% | 41 k | 5/5 |
| IQL → PPO + KL | 2.43% | 1.51% ± 0.27 | 3.20% | 13 k | 5/5 |
| IQL + HUBL → PPO + KL | 2.44% | 1.61% ± 0.13 | 2.72% | 54 k | 5/5 |

Variants that use Sum of Squares offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its 12,800 steps in their step counts; HEPO's heuristic rollouts are half of its PPO steps. 'Beat' is measured on the 64 evaluation episodes of the learning curves.
