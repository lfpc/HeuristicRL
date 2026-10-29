# Online bin packing: guided by Best Fit

Wasted bin capacity (%) (lower is better), greedy policies on 256 fixed episodes
(B = 12, item sizes {3, 4}, p = {2/3, 1/3}, 200 items).

| Heuristic | Metric |
|---|---|
| Best Fit (guide) | 9.68% ± 0.99 |
| Sum of Squares | 2.36% ± 0.66 |
| Random | 13.56% ± 2.70 |

| Variant | Initial | Final | Worst during training | Steps to beat Best Fit (median) | Seeds beating Best Fit |
|---|---|---|---|---|---|
| PPO from scratch | 58.87% | 2.48% ± 0.85 | 14.05% | 573 k | 5/5 |
| BC → PPO (naive) | 9.86% | 3.62% ± 1.48 | 11.53% | 812 k | 5/5 |
| BC → PPO + KL (decay 50%) | 9.86% | 1.63% ± 0.23 | 13.30% | 771 k | 5/5 |
| BC → PPO + KL (decay 20%) | 9.86% | 2.63% ± 1.11 | 13.42% | 648 k | 5/5 |
| DAgger → PPO (naive) | 9.86% | 2.22% ± 0.98 | 15.29% | 443 k | 5/5 |
| DAgger → PPO + KL (decay 50%) | 9.86% | 3.67% ± 1.97 | 11.44% | 832 k | 5/5 |
| HuRL (from scratch) | 58.87% | 1.66% ± 0.31 | 23.46% | 156 k | 5/5 |
| BC → HuRL | 9.86% | 1.70% ± 0.31 | 15.37% | 218 k | 5/5 |
| HEPO (from scratch) | 58.87% | 9.68% ± 0.00 | 10.02% | 328 k | 4/5 |
| IQL → PPO + KL | 9.86% | 2.02% ± 0.60 | 10.87% | 566 k | 5/5 |
| IQL + HUBL → PPO + KL | 9.86% | 2.00% ± 0.18 | 9.88% | 545 k | 5/5 |

Variants that use Best Fit offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its 12,800 steps in their step counts; HEPO's heuristic rollouts are half of its PPO steps. 'Beat' is measured on the 64 evaluation episodes of the learning curves.
