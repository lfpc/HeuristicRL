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
| PPO from scratch | 67.89% | 1.67% ± 0.25 | 54.44% | 246 k | 10/10 |
| BC → PPO (naive) | 4.72% | 1.63% ± 0.15 | 8.53% | 125 k | 10/10 |
| BC → PPO + KL (decay 50%) | 4.72% | 1.64% ± 0.12 | 4.81% | 125 k | 10/10 |
| BC → PPO + KL (decay 20%) | 4.72% | 1.62% ± 0.21 | 4.80% | 125 k | 10/10 |
| DAgger → PPO (naive) | 4.55% | 1.72% ± 0.21 | 6.34% | 105 k | 10/10 |
| DAgger → PPO + KL (decay 50%) | 4.55% | 1.86% ± 0.13 | 5.11% | 392 k | 10/10 |
| HuRL (from scratch) | 67.89% | 1.63% ± 0.24 | 57.47% | 105 k | 10/10 |
| BC → HuRL | 4.72% | 1.53% ± 0.10 | 5.81% | 95 k | 10/10 |
| HEPO (from scratch) | 67.89% | 3.02% ± 0.21 | 10.24% | 102 k | 10/10 |
| IQL → PPO + KL | 6.93% | 1.61% ± 0.23 | 6.94% | 146 k | 10/10 |
| IQL + HUBL → PPO + KL | 4.59% | 1.55% ± 0.18 | 4.71% | 84 k | 10/10 |
| Kickstarting (from scratch) | 67.89% | 1.78% ± 0.07 | 4.84% | 61 k | 10/10 |
| BC → PPO + heuristic anchor | 4.72% | 3.12% ± 0.17 | 4.94% | 84 k | 10/10 |
| BC → PPO + heuristic anchor, 4× batch | 4.72% | 3.04% ± 0.13 | 4.73% | 177 k | 10/10 |
| IQL + HUBL → PPO + heuristic anchor | 4.58% | 2.51% ± 0.29 | 4.79% | 54 k | 10/10 |
| Residual policy on heuristic | 4.55% | 2.38% ± 0.91 | 4.56% | 809 k | 8/10 |
| Jump-Start RL | 67.89% | 1.54% ± 0.10 | 55.12% | 276 k | 10/10 |
| Rollout-DAgger → PPO + KL | 1.44% | 1.48% ± 0.24 | 2.27% | 13 k | 10/10 |
| Rollout-DAgger → PPO + held KL | 1.44% | 1.30% ± 0.04 | 1.67% | 13 k | 10/10 |
| IQL offline → online | 4.61% | 3.67% ± 0.09 | 4.81% | 74 k | 9/10 |
| DQfD | 4.55% | 3.68% ± 0.01 | 4.56% | 381 k | 3/10 |
| Imitation, annealed by 25% (Crafter) | 67.89% | 1.66% ± 0.23 | 4.91% | 51 k | 10/10 |
| Imitation, constant 0.1 (Crafter) | 67.89% | 3.08% ± 0.11 | 25.77% | 102 k | 10/10 |
| Residual prior β = 2 (Crafter) | 4.55% | 1.52% ± 0.08 | 5.70% | 61 k | 10/10 |
| Residual prior β = 4 (Crafter) | 4.55% | 1.44% ± 0.07 | 4.55% | 297 k | 10/10 |
| Residual prior β = 8 (Crafter) | 4.55% | 2.34% ± 0.69 | 4.55% | 840 k | 9/10 |
| Residual prior β = 4, fading | 4.55% | 1.54% ± 0.25 | 4.55% | 164 k | 10/10 |
| DQfD, fading | 4.55% | 1.42% ± 0.22 | 4.65% | 730 k | 10/10 |
| IQL offline → online, fading | 4.61% | 3.51% ± 0.22 | 4.90% | 207 k | 10/10 |
| BC → PPO + annealed imitation | 4.72% | 1.75% ± 0.18 | 4.94% | 115 k | 10/10 |
| IQL + HUBL → PPO + annealed imitation | 4.58% | 1.50% ± 0.21 | 4.88% | 125 k | 10/10 |
| IQL offline → online, fading, normalised AWR | 49.46% | 1.25% ± 0.07 | 49.37% | 330 k | 10/10 |
| IQL offline → online, normalised AWR, held | 49.46% | 4.10% ± 0.70 | 49.34% | 340 k | 10/10 |
| IQL offline → online, fading, normalised AWR, γ = 0.99 | 51.16% | 13.76% ± 0.52 | 52.27% | never | 0/10 |
| Rollout-DAgger (no test) → PPO + KL | 2.05% | 1.38% ± 0.12 | 2.18% | 13 k | 10/10 |
| BC → PPO + Q-filtered imitation | 4.72% | 2.50% ± 0.30 | 4.79% | 95 k | 10/10 |
| IQL + HUBL → PPO + Q-filtered imitation | 4.58% | 2.23% ± 0.25 | 4.65% | 64 k | 10/10 |
| Performance-gated imitation (from scratch) | 67.89% | 1.62% ± 0.25 | 5.40% | 44 k | 10/10 |
| BC → PPO + performance-gated imitation | 4.72% | 1.59% ± 0.12 | 5.15% | 99 k | 10/10 |
| IQL + HUBL → PPO + performance-gated imitation | 4.58% | 1.44% ± 0.07 | 4.90% | 108 k | 10/10 |

Variants that use Static threshold offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its 12,800 steps in their step counts; HEPO's heuristic rollouts are half of its PPO steps. 'Beat' is measured on the 64 evaluation episodes of the learning curves.
