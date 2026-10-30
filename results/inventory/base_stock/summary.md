# Lost-sales inventory: guided by Base stock

Mean cost per period (lower is better), greedy policies on 256 fixed episodes
(Poisson(5) demand, lead time 3, holding 1, lost sales 9, orders 0..15, 100 periods).

| Heuristic | Metric |
|---|---|
| Base stock (guide) | 6.94 ± 0.81 |
| Capped base stock | 6.70 ± 0.76 |
| Constant | 15.11 ± 7.87 |
| Random | 129.00 ± 25.89 |

| Variant | Initial | Final | Worst during training | Steps to beat Base stock (median) | Seeds beating Base stock |
|---|---|---|---|---|---|
| PPO from scratch | 107.61 | 6.71 ± 0.03 | 35.07 | 133 k | 10/10 |
| BC → PPO (naive) | 6.75 | 6.71 ± 0.06 | 7.01 | 6 k | 10/10 |
| BC → PPO + KL (decay 50%) | 6.75 | 6.72 ± 0.05 | 6.93 | 6 k | 10/10 |
| BC → PPO + KL (decay 20%) | 6.75 | 6.72 ± 0.04 | 6.94 | 6 k | 10/10 |
| DAgger → PPO (naive) | 6.99 | 6.68 ± 0.02 | 6.97 | 17 k | 10/10 |
| DAgger → PPO + KL (decay 50%) | 6.99 | 6.70 ± 0.03 | 7.01 | 27 k | 10/10 |
| HuRL (from scratch) | 107.61 | 6.70 ± 0.03 | 29.87 | 140 k | 10/10 |
| BC → HuRL | 6.75 | 6.74 ± 0.04 | 7.12 | 6 k | 10/10 |
| HEPO (from scratch) | 107.61 | 6.77 ± 0.03 | 15.94 | 143 k | 10/10 |
| IQL → PPO + KL | 6.98 | 6.72 ± 0.03 | 6.99 | 6 k | 10/10 |
| IQL + HUBL → PPO + KL | 6.98 | 6.72 ± 0.02 | 6.99 | 6 k | 10/10 |
| Kickstarting (from scratch) | 107.61 | 6.73 ± 0.03 | 10.48 | 61 k | 10/10 |
| BC → PPO + heuristic anchor | 6.75 | 6.90 ± 0.01 | 6.99 | 6 k | 10/10 |
| BC → PPO + heuristic anchor, 4× batch | 6.75 | 6.90 ± 0.01 | 6.95 | 6 k | 10/10 |
| IQL + HUBL → PPO + heuristic anchor | 6.99 | 6.93 ± 0.00 | 7.00 | 6 k | 10/10 |
| Residual policy on heuristic | 6.98 | 6.73 ± 0.04 | 7.00 | 92 k | 10/10 |
| Jump-Start RL | 107.61 | 6.91 ± 0.16 | 213.51 | 778 k | 9/10 |
| Rollout-DAgger → PPO + KL | 6.88 | 6.69 ± 0.03 | 6.92 | 6 k | 10/10 |
| Rollout-DAgger → PPO + held KL | 6.88 | 6.77 ± 0.03 | 6.90 | 6 k | 10/10 |
| IQL offline → online | 6.98 | 6.94 ± 0.00 | 6.98 | 6 k | 10/10 |
| DQfD | 6.92 | 6.94 ± 0.00 | 7.38 | 6 k | 10/10 |
| Imitation, annealed by 25% (Crafter) | 107.61 | 6.68 ± 0.02 | 10.58 | 61 k | 10/10 |
| Imitation, constant 0.1 (Crafter) | 107.61 | 6.91 ± 0.01 | 26.66 | 102 k | 10/10 |
| Residual prior β = 2 (Crafter) | 6.98 | 6.76 ± 0.07 | 7.40 | 61 k | 10/10 |
| Residual prior β = 4 (Crafter) | 6.98 | 6.75 ± 0.03 | 7.03 | 82 k | 10/10 |
| Residual prior β = 8 (Crafter) | 6.98 | 6.83 ± 0.06 | 7.02 | 195 k | 10/10 |
| Residual prior β = 4, fading | 6.98 | 6.71 ± 0.02 | 7.08 | 102 k | 10/10 |
| DQfD, fading | 6.92 | 7.37 ± 0.38 | 10.13 | 6 k | 10/10 |
| IQL offline → online, fading | 6.98 | 6.94 ± 0.00 | 6.98 | 6 k | 10/10 |
| BC → PPO + annealed imitation | 6.75 | 6.69 ± 0.03 | 6.96 | 6 k | 10/10 |
| IQL + HUBL → PPO + annealed imitation | 6.99 | 6.72 ± 0.04 | 6.99 | 6 k | 10/10 |
| IQL offline → online, fading, normalised AWR | 7.13 | 6.66 ± 0.02 | 8.34 | 119 k | 10/10 |
| IQL offline → online, normalised AWR, held | 7.13 | 6.88 ± 0.04 | 8.33 | 99 k | 10/10 |
| IQL offline → online, fading, normalised AWR, γ = 0.99 | 7.19 | 6.67 ± 0.01 | 8.00 | 99 k | 10/10 |
| Rollout-DAgger (no test) → PPO + KL | 6.76 | 6.73 ± 0.03 | 6.87 | 6 k | 10/10 |
| BC → PPO + Q-filtered imitation | 6.75 | 6.89 ± 0.03 | 7.13 | 6 k | 10/10 |
| IQL + HUBL → PPO + Q-filtered imitation | 6.99 | 6.93 ± 0.03 | 7.01 | 6 k | 10/10 |
| Performance-gated imitation (from scratch) | 107.61 | 6.70 ± 0.03 | 10.49 | 64 k | 10/10 |
| BC → PPO + performance-gated imitation | 6.75 | 6.70 ± 0.03 | 6.93 | 6 k | 10/10 |
| IQL + HUBL → PPO + performance-gated imitation | 6.99 | 6.70 ± 0.04 | 7.02 | 6 k | 10/10 |

Variants that use Base stock offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its 6,400 steps in their step counts; HEPO's heuristic rollouts are half of its PPO steps. 'Beat' is measured on the 64 evaluation episodes of the learning curves.
