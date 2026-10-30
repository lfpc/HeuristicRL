# Load balancing: guided by Shortest expected delay

Mean sojourn time (lower is better), greedy policies on 256 fixed episodes
(4 FIFO servers with speeds (0.4, 0.8, 1.2, 1.6), Exp(1) job sizes, load 0.85, 200 jobs).

| Heuristic | Metric |
|---|---|
| Shortest expected delay (guide) | 1.73 ± 0.70 |
| Least work left | 1.96 ± 0.71 |
| Random | 3.99 ± 1.28 |

| Variant | Initial | Final | Worst during training | Steps to beat Shortest expected delay (median) | Seeds beating Shortest expected delay |
|---|---|---|---|---|---|
| PPO from scratch | 53.14 | 1.64 ± 0.02 | 14.38 | 174 k | 10/10 |
| BC → PPO (naive) | 1.76 | 1.65 ± 0.02 | 1.77 | 13 k | 10/10 |
| BC → PPO + KL (decay 50%) | 1.76 | 1.67 ± 0.01 | 1.77 | 13 k | 10/10 |
| BC → PPO + KL (decay 20%) | 1.76 | 1.66 ± 0.03 | 1.77 | 13 k | 10/10 |
| DAgger → PPO (naive) | 1.76 | 1.67 ± 0.02 | 1.77 | 13 k | 10/10 |
| DAgger → PPO + KL (decay 50%) | 1.76 | 1.69 ± 0.02 | 1.77 | 13 k | 10/10 |
| HuRL (from scratch) | 53.14 | 1.63 ± 0.02 | 9.62 | 617 k | 10/10 |
| BC → HuRL | 1.76 | 1.66 ± 0.02 | 1.78 | 13 k | 10/10 |
| HEPO (from scratch) | 53.14 | 1.72 ± 0.00 | 4.12 | 143 k | 10/10 |
| IQL → PPO + KL | 1.75 | 1.67 ± 0.02 | 1.76 | 13 k | 10/10 |
| IQL + HUBL → PPO + KL | 1.75 | 1.67 ± 0.02 | 1.77 | 13 k | 10/10 |
| Kickstarting (from scratch) | 53.14 | 1.69 ± 0.02 | 1.81 | 92 k | 10/10 |
| BC → PPO + heuristic anchor | 1.76 | 1.72 ± 0.00 | 1.77 | 13 k | 10/10 |
| BC → PPO + heuristic anchor, 4× batch | 1.76 | 1.72 ± 0.00 | 1.76 | 13 k | 10/10 |
| IQL + HUBL → PPO + heuristic anchor | 1.76 | 1.72 ± 0.00 | 1.77 | 13 k | 10/10 |
| Residual policy on heuristic | 1.76 | 1.65 ± 0.03 | 1.79 | 287 k | 10/10 |
| Jump-Start RL | 53.14 | 1.65 ± 0.04 | 44.42 | 430 k | 10/10 |
| Rollout-DAgger → PPO + KL | 1.70 | 1.60 ± 0.01 | 1.71 | 13 k | 10/10 |
| Rollout-DAgger → PPO + held KL | 1.70 | 1.65 ± 0.01 | 1.71 | 13 k | 10/10 |
| IQL offline → online | 1.76 | 1.73 ± 0.00 | 1.76 | 13 k | 10/10 |
| DQfD | 1.76 | 1.73 ± 0.01 | 1.86 | 23 k | 10/10 |
| Imitation, annealed by 25% (Crafter) | 53.14 | 1.68 ± 0.03 | 1.81 | 92 k | 10/10 |
| Imitation, constant 0.1 (Crafter) | 53.14 | 1.72 ± 0.01 | 2.29 | 102 k | 10/10 |
| Residual prior β = 2 (Crafter) | 1.76 | 1.63 ± 0.01 | 1.77 | 92 k | 10/10 |
| Residual prior β = 4 (Crafter) | 1.76 | 1.68 ± 0.04 | 1.79 | 440 k | 10/10 |
| Residual prior β = 8 (Crafter) | 1.76 | 1.69 ± 0.04 | 1.77 | 573 k | 9/10 |
| Residual prior β = 4, fading | 1.76 | 1.63 ± 0.02 | 1.78 | 297 k | 10/10 |
| DQfD, fading | 1.76 | 1.72 ± 0.06 | 3.44 | 23 k | 10/10 |
| IQL offline → online, fading | 1.76 | 1.71 ± 0.00 | 1.76 | 13 k | 10/10 |
| BC → PPO + annealed imitation | 1.76 | 1.69 ± 0.01 | 1.77 | 13 k | 10/10 |
| IQL + HUBL → PPO + annealed imitation | 1.76 | 1.67 ± 0.03 | 1.77 | 13 k | 10/10 |
| IQL offline → online, fading, normalised AWR | 2.24 | 1.58 ± 0.01 | 3.81 | 95 k | 10/10 |
| IQL offline → online, normalised AWR, held | 2.24 | 1.66 ± 0.02 | 3.66 | 105 k | 10/10 |
| IQL offline → online, fading, normalised AWR, γ = 0.99 | 1.99 | 1.57 ± 0.01 | 2.45 | 95 k | 10/10 |
| Rollout-DAgger (no test) → PPO + KL | 1.59 | 1.59 ± 0.02 | 1.65 | 13 k | 10/10 |
| BC → PPO + Q-filtered imitation | 1.76 | 1.72 ± 0.00 | 1.77 | 13 k | 10/10 |
| IQL + HUBL → PPO + Q-filtered imitation | 1.76 | 1.72 ± 0.00 | 1.77 | 13 k | 10/10 |
| Performance-gated imitation (from scratch) | 53.14 | 1.65 ± 0.03 | 1.81 | 88 k | 10/10 |
| BC → PPO + performance-gated imitation | 1.76 | 1.66 ± 0.02 | 1.77 | 13 k | 10/10 |
| IQL + HUBL → PPO + performance-gated imitation | 1.76 | 1.67 ± 0.04 | 1.78 | 13 k | 10/10 |

Variants that use Shortest expected delay offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its 12,800 steps in their step counts; HEPO's heuristic rollouts are half of its PPO steps. 'Beat' is measured on the 64 evaluation episodes of the learning curves.
