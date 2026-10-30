# Online bin packing: guided by EoH-evolved

Wasted bin capacity (%) (lower is better), greedy policies on 256 fixed episodes
(B = 12, item sizes {3, 4}, p = {2/3, 1/3}, 200 items).

| Heuristic | Metric |
|---|---|
| Best Fit | 9.68% ± 0.99 |
| Sum of Squares | 2.36% ± 0.66 |
| EoH-evolved (guide) | 11.48% ± 1.67 |
| Random | 13.56% ± 2.70 |

| Variant | Initial | Final | Worst during training | Steps to beat EoH-evolved (median) | Seeds beating EoH-evolved |
|---|---|---|---|---|---|
| PPO from scratch | 55.49% | 3.59% ± 2.14 | 15.67% | 20 k | 10/10 |
| BC → PPO (naive) | 12.03% | 2.08% ± 1.17 | 12.30% | 33 k | 10/10 |
| BC → PPO + KL (decay 50%) | 12.03% | 1.32% ± 0.14 | 12.40% | 74 k | 10/10 |
| BC → PPO + KL (decay 20%) | 12.03% | 1.77% ± 0.48 | 12.48% | 74 k | 10/10 |
| DAgger → PPO (naive) | 15.11% | 1.96% ± 0.63 | 12.49% | 33 k | 10/10 |
| DAgger → PPO + KL (decay 50%) | 15.11% | 1.62% ± 0.42 | 15.44% | 105 k | 10/10 |
| HuRL (from scratch) | 55.49% | 1.78% ± 0.47 | 24.10% | 33 k | 10/10 |
| BC → HuRL | 12.03% | 1.72% ± 0.35 | 12.13% | 33 k | 10/10 |
| HEPO (from scratch) | 55.49% | 11.28% ± 0.17 | 72.21% | 61 k | 10/10 |
| IQL → PPO + KL | 10.58% | 1.70% ± 0.33 | 10.82% | 13 k | 10/10 |
| IQL + HUBL → PPO + KL | 10.55% | 1.61% ± 0.26 | 10.89% | 13 k | 10/10 |
| Kickstarting (from scratch) | 55.49% | 1.44% ± 0.30 | 54.66% | 143 k | 10/10 |
| BC → PPO + heuristic anchor | 12.03% | 4.25% ± 0.48 | 12.51% | 54 k | 10/10 |
| BC → PPO + heuristic anchor, 4× batch | 12.03% | 3.49% ± 0.39 | 12.03% | 177 k | 10/10 |
| IQL + HUBL → PPO + heuristic anchor | 11.14% | 3.88% ± 0.62 | 12.41% | 13 k | 10/10 |
| Residual policy on heuristic | 11.67% | 3.38% ± 2.24 | 15.52% | 92 k | 10/10 |
| Jump-Start RL | 55.49% | 1.47% ± 0.35 | 10.14% | 20 k | 10/10 |
| Rollout-DAgger → PPO + KL | 8.94% | 1.70% ± 0.34 | 10.13% | 13 k | 10/10 |
| Rollout-DAgger → PPO + held KL | 8.94% | 3.79% ± 0.40 | 11.73% | 13 k | 10/10 |
| IQL offline → online | 11.51% | 11.37% ± 0.23 | 12.22% | 13 k | 10/10 |
| DQfD | 11.62% | 11.62% ± 0.46 | 13.83% | 23 k | 10/10 |
| Imitation, annealed by 25% (Crafter) | 55.49% | 1.48% ± 0.30 | 54.04% | 102 k | 10/10 |
| Imitation, constant 0.1 (Crafter) | 55.49% | 3.88% ± 0.50 | 26.01% | 20 k | 10/10 |
| Residual prior β = 2 (Crafter) | 11.67% | 2.40% ± 1.05 | 10.97% | 20 k | 10/10 |
| Residual prior β = 4 (Crafter) | 11.67% | 2.87% ± 1.30 | 12.14% | 20 k | 10/10 |
| Residual prior β = 8 (Crafter) | 11.67% | 3.84% ± 1.22 | 13.20% | 61 k | 10/10 |
| Residual prior β = 4, fading | 11.67% | 3.27% ± 1.32 | 12.29% | 20 k | 10/10 |
| DQfD, fading | 11.62% | 3.96% ± 1.51 | 22.80% | 23 k | 10/10 |
| IQL offline → online, fading | 11.51% | 10.60% ± 0.26 | 11.65% | 13 k | 10/10 |
| BC → PPO + annealed imitation | 12.03% | 1.50% ± 0.35 | 14.64% | 95 k | 10/10 |
| IQL + HUBL → PPO + annealed imitation | 11.14% | 1.59% ± 0.33 | 14.83% | 13 k | 10/10 |
| IQL offline → online, fading, normalised AWR | 3.98% | 3.30% ± 1.29 | 8.33% | 13 k | 10/10 |
| IQL offline → online, normalised AWR, held | 3.98% | 2.00% ± 0.62 | 5.54% | 13 k | 10/10 |
| IQL offline → online, fading, normalised AWR, γ = 0.99 | 4.06% | 1.42% ± 0.20 | 3.60% | 13 k | 10/10 |
| Rollout-DAgger (no test) → PPO + KL | 3.87% | 1.62% ± 0.29 | 8.07% | 13 k | 10/10 |
| BC → PPO + Q-filtered imitation | 12.03% | 3.65% ± 0.72 | 12.38% | 95 k | 10/10 |
| IQL + HUBL → PPO + Q-filtered imitation | 11.14% | 3.32% ± 0.59 | 12.54% | 13 k | 10/10 |
| Performance-gated imitation (from scratch) | 55.49% | 1.56% ± 0.42 | 50.26% | 155 k | 10/10 |
| BC → PPO + performance-gated imitation | 12.03% | 1.66% ± 0.54 | 14.69% | 122 k | 10/10 |
| IQL + HUBL → PPO + performance-gated imitation | 11.14% | 1.71% ± 0.33 | 14.78% | 13 k | 10/10 |

Variants that use EoH-evolved offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its 12,800 steps in their step counts; HEPO's heuristic rollouts are half of its PPO steps. 'Beat' is measured on the 64 evaluation episodes of the learning curves.
