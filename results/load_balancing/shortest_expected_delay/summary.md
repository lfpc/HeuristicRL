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
| PPO from scratch | 56.70 | 1.64 ± 0.02 | 12.35 | 205 k | 5/5 |
| BC → PPO (naive) | 1.76 | 1.66 ± 0.02 | 1.77 | 13 k | 5/5 |
| BC → PPO + KL (decay 50%) | 1.76 | 1.67 ± 0.02 | 1.77 | 13 k | 5/5 |
| BC → PPO + KL (decay 20%) | 1.76 | 1.67 ± 0.02 | 1.77 | 13 k | 5/5 |
| DAgger → PPO (naive) | 1.76 | 1.67 ± 0.01 | 1.77 | 13 k | 5/5 |
| DAgger → PPO + KL (decay 50%) | 1.76 | 1.69 ± 0.02 | 1.77 | 13 k | 5/5 |
| HuRL (from scratch) | 56.70 | 1.62 ± 0.01 | 8.29 | 504 k | 5/5 |
| BC → HuRL | 1.76 | 1.65 ± 0.02 | 1.78 | 13 k | 5/5 |
| HEPO (from scratch) | 56.70 | 1.72 ± 0.00 | 4.51 | 143 k | 5/5 |
| IQL → PPO + KL | 1.75 | 1.68 ± 0.01 | 1.76 | 13 k | 5/5 |
| IQL + HUBL → PPO + KL | 1.75 | 1.68 ± 0.01 | 1.78 | 13 k | 5/5 |

Variants that use Shortest expected delay offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its 12,800 steps in their step counts; HEPO's heuristic rollouts are half of its PPO steps. 'Beat' is measured on the 64 evaluation episodes of the learning curves.
