# Online bin packing: guided by Sum of Squares

Wasted bin capacity (%) (lower is better), greedy policies on 256 fixed episodes
(B = 12, item sizes {3, 4}, p = {2/3, 1/3}, 200 items).

| Heuristic | Metric |
|---|---|
| Best Fit | 9.68% ± 0.99 |
| Sum of Squares (guide) | 2.36% ± 0.66 |
| EoH-evolved | 11.48% ± 1.67 |
| Random | 13.56% ± 2.70 |

| Variant | Initial | Final | Worst during training | Steps to beat Sum of Squares (median) | Seeds beating Sum of Squares |
|---|---|---|---|---|---|
| PPO from scratch | 55.49% | 3.59% ± 2.14 | 15.67% | 1372 k | 6/10 |
| BC → PPO (naive) | 2.47% | 1.89% ± 0.25 | 4.32% | 23 k | 10/10 |
| BC → PPO + KL (decay 50%) | 2.47% | 1.72% ± 0.17 | 2.83% | 33 k | 10/10 |
| BC → PPO + KL (decay 20%) | 2.47% | 1.76% ± 0.21 | 3.26% | 33 k | 10/10 |
| DAgger → PPO (naive) | 2.49% | 1.94% ± 0.27 | 5.12% | 44 k | 10/10 |
| DAgger → PPO + KL (decay 50%) | 2.49% | 1.69% ± 0.13 | 2.64% | 525 k | 10/10 |
| HuRL (from scratch) | 55.49% | 2.41% ± 0.72 | 15.41% | 1180 k | 9/10 |
| BC → HuRL | 2.47% | 2.22% ± 0.33 | 6.38% | 23 k | 10/10 |
| HEPO (from scratch) | 55.49% | 2.39% ± 0.05 | 15.71% | 41 k | 10/10 |
| IQL → PPO + KL | 2.43% | 1.68% ± 0.16 | 2.77% | 136 k | 10/10 |
| IQL + HUBL → PPO + KL | 2.45% | 1.51% ± 0.18 | 3.45% | 54 k | 10/10 |
| Kickstarting (from scratch) | 55.49% | 1.43% ± 0.27 | 4.79% | 819 k | 10/10 |
| BC → PPO + heuristic anchor | 2.47% | 2.22% ± 0.21 | 4.18% | 320 k | 10/10 |
| BC → PPO + heuristic anchor, 4× batch | 2.47% | 2.05% ± 0.22 | 3.91% | 13 k | 9/10 |
| IQL + HUBL → PPO + heuristic anchor | 2.44% | 2.01% ± 0.14 | 7.56% | 617 k | 10/10 |
| Residual policy on heuristic | 2.46% | 2.55% ± 0.41 | 12.56% | 840 k | 5/10 |
| Jump-Start RL | 55.49% | 8.63% ± 5.57 | 46.39% | never | 0/10 |
| Rollout-DAgger → PPO + KL | 2.50% | 1.59% ± 0.20 | 2.63% | 125 k | 10/10 |
| Rollout-DAgger → PPO + held KL | 2.50% | 1.96% ± 0.12 | 3.28% | 340 k | 10/10 |
| IQL offline → online | 2.45% | 2.34% ± 0.06 | 2.48% | 33 k | 7/10 |
| DQfD | 2.46% | 2.37% ± 0.03 | 2.46% | never | 0/10 |
| Imitation, annealed by 25% (Crafter) | 55.49% | 1.86% ± 0.10 | 5.50% | 625 k | 10/10 |
| Imitation, constant 0.1 (Crafter) | 55.49% | 2.14% ± 0.16 | 17.01% | 829 k | 10/10 |
| Residual prior β = 2 (Crafter) | 2.46% | 2.10% ± 0.39 | 10.17% | 829 k | 10/10 |
| Residual prior β = 4 (Crafter) | 2.46% | 2.23% ± 0.23 | 9.61% | 1075 k | 10/10 |
| Residual prior β = 8 (Crafter) | 2.46% | 2.69% ± 0.27 | 6.68% | 1167 k | 4/10 |
| Residual prior β = 4, fading | 2.46% | 2.32% ± 0.65 | 9.70% | 758 k | 9/10 |
| DQfD, fading | 2.46% | 3.13% ± 1.15 | 14.05% | 1160 k | 10/10 |
| IQL offline → online, fading | 2.45% | 2.36% ± 0.00 | 2.48% | 13 k | 3/10 |
| BC → PPO + annealed imitation | 2.47% | 1.87% ± 0.14 | 3.50% | 218 k | 10/10 |
| IQL + HUBL → PPO + annealed imitation | 2.44% | 1.85% ± 0.17 | 3.67% | 105 k | 10/10 |
| IQL offline → online, fading, normalised AWR | 2.79% | 2.80% ± 1.13 | 6.67% | 33 k | 10/10 |
| IQL offline → online, normalised AWR, held | 2.79% | 1.33% ± 0.24 | 2.64% | 33 k | 10/10 |
| IQL offline → online, fading, normalised AWR, γ = 0.99 | 2.54% | 1.15% ± 0.09 | 3.09% | 23 k | 10/10 |
| Rollout-DAgger (no test) → PPO + KL | 3.47% | 2.05% ± 0.16 | 4.43% | 740 k | 10/10 |
| BC → PPO + Q-filtered imitation | 2.47% | 2.18% ± 0.17 | 4.60% | 33 k | 10/10 |
| IQL + HUBL → PPO + Q-filtered imitation | 2.44% | 2.03% ± 0.13 | 11.57% | 627 k | 10/10 |
| Performance-gated imitation (from scratch) | 55.49% | 1.86% ± 0.33 | 7.66% | 650 k | 10/10 |
| BC → PPO + performance-gated imitation | 2.47% | 2.01% ± 0.16 | 4.53% | 112 k | 10/10 |
| IQL + HUBL → PPO + performance-gated imitation | 2.44% | 2.07% ± 0.11 | 7.50% | 57 k | 10/10 |

Variants that use Sum of Squares offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its 12,800 steps in their step counts; HEPO's heuristic rollouts are half of its PPO steps. 'Beat' is measured on the 64 evaluation episodes of the learning curves.
