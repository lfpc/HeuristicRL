# Online bin packing: guided by Best Fit

Wasted bin capacity (%) (lower is better), greedy policies on 256 fixed episodes
(B = 12, item sizes {3, 4}, p = {2/3, 1/3}, 200 items).

| Heuristic | Metric |
|---|---|
| Best Fit (guide) | 9.68% ± 0.99 |
| Sum of Squares | 2.36% ± 0.66 |
| EoH-evolved | 11.48% ± 1.67 |
| Random | 13.56% ± 2.70 |

| Variant | Initial | Final | Worst during training | Steps to beat Best Fit (median) | Seeds beating Best Fit |
|---|---|---|---|---|---|
| PPO from scratch | 55.49% | 3.59% ± 2.14 | 15.67% | 635 k | 10/10 |
| BC → PPO (naive) | 9.86% | 3.53% ± 2.59 | 12.85% | 730 k | 10/10 |
| BC → PPO + KL (decay 50%) | 9.86% | 2.17% ± 0.86 | 12.06% | 740 k | 10/10 |
| BC → PPO + KL (decay 20%) | 9.86% | 2.30% ± 0.70 | 15.23% | 535 k | 10/10 |
| DAgger → PPO (naive) | 9.86% | 2.50% ± 0.72 | 19.88% | 709 k | 10/10 |
| DAgger → PPO + KL (decay 50%) | 9.86% | 1.93% ± 0.51 | 13.78% | 842 k | 10/10 |
| HuRL (from scratch) | 55.49% | 1.73% ± 0.33 | 22.42% | 177 k | 10/10 |
| BC → HuRL | 9.86% | 1.57% ± 0.26 | 24.08% | 228 k | 10/10 |
| HEPO (from scratch) | 55.49% | 9.68% ± 0.00 | 9.96% | 266 k | 8/10 |
| IQL → PPO + KL | 9.86% | 1.93% ± 0.21 | 9.90% | 545 k | 10/10 |
| IQL + HUBL → PPO + KL | 9.86% | 2.09% ± 0.27 | 12.80% | 566 k | 10/10 |
| Kickstarting (from scratch) | 55.49% | 2.80% ± 2.39 | 18.67% | 901 k | 9/10 |
| BC → PPO + heuristic anchor | 9.86% | 7.35% ± 2.02 | 10.89% | 975 k | 7/10 |
| BC → PPO + heuristic anchor, 4× batch | 9.86% | 3.79% ± 0.60 | 10.02% | 545 k | 10/10 |
| IQL + HUBL → PPO + heuristic anchor | 9.86% | 5.29% ± 1.63 | 10.48% | 1078 k | 10/10 |
| Residual policy on heuristic | 9.86% | 3.74% ± 2.40 | 11.33% | 369 k | 9/10 |
| Jump-Start RL | 55.49% | 1.81% ± 0.69 | 14.12% | 358 k | 10/10 |
| Rollout-DAgger → PPO + KL | 8.15% | 1.96% ± 0.23 | 9.58% | 13 k | 10/10 |
| Rollout-DAgger → PPO + held KL | 8.15% | 3.01% ± 0.44 | 9.86% | 13 k | 10/10 |
| IQL offline → online | 9.86% | 9.69% ± 0.01 | 9.88% | 1324 k | 3/10 |
| DQfD | 9.86% | 9.68% ± 0.00 | 9.86% | never | 0/10 |
| Imitation, annealed by 25% (Crafter) | 55.49% | 2.11% ± 0.86 | 14.21% | 512 k | 10/10 |
| Imitation, constant 0.1 (Crafter) | 55.49% | 6.49% ± 2.63 | 15.63% | 983 k | 9/10 |
| Residual prior β = 2 (Crafter) | 9.86% | 2.84% ± 1.34 | 13.62% | 440 k | 10/10 |
| Residual prior β = 4 (Crafter) | 9.86% | 2.79% ± 0.82 | 12.17% | 461 k | 10/10 |
| Residual prior β = 8 (Crafter) | 9.86% | 5.05% ± 2.76 | 12.82% | 471 k | 9/10 |
| Residual prior β = 4, fading | 9.86% | 2.50% ± 0.98 | 14.05% | 389 k | 10/10 |
| DQfD, fading | 9.86% | 3.34% ± 1.11 | 63.71% | 822 k | 10/10 |
| IQL offline → online, fading | 9.86% | 9.68% ± 0.01 | 9.86% | 1364 k | 1/10 |
| BC → PPO + annealed imitation | 9.86% | 2.00% ± 0.23 | 17.56% | 484 k | 10/10 |
| IQL + HUBL → PPO + annealed imitation | 9.86% | 2.29% ± 0.72 | 14.81% | 556 k | 10/10 |
| IQL offline → online, fading, normalised AWR | 10.02% | 3.11% ± 1.04 | 8.17% | 13 k | 10/10 |
| IQL offline → online, normalised AWR, held | 10.02% | 3.95% ± 0.89 | 8.58% | 13 k | 10/10 |
| IQL offline → online, fading, normalised AWR, γ = 0.99 | 7.85% | 2.11% ± 1.65 | 4.12% | 13 k | 10/10 |
| Rollout-DAgger (no test) → PPO + KL | 2.89% | 1.18% ± 0.09 | 6.05% | 13 k | 10/10 |
| BC → PPO + Q-filtered imitation | 9.86% | 3.48% ± 0.68 | 11.46% | 535 k | 10/10 |
| IQL + HUBL → PPO + Q-filtered imitation | 9.86% | 3.26% ± 0.60 | 13.29% | 371 k | 10/10 |
| Performance-gated imitation (from scratch) | 55.49% | 2.35% ± 0.87 | 17.27% | 454 k | 10/10 |
| BC → PPO + performance-gated imitation | 9.86% | 3.07% ± 1.59 | 12.69% | 876 k | 10/10 |
| IQL + HUBL → PPO + performance-gated imitation | 9.86% | 2.69% ± 1.17 | 16.22% | 364 k | 10/10 |

Variants that use Best Fit offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its 12,800 steps in their step counts; HEPO's heuristic rollouts are half of its PPO steps. 'Beat' is measured on the 64 evaluation episodes of the learning curves.
