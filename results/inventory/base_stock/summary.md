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
| PPO from scratch | 99.14 | 6.70 ± 0.02 | 38.05 | 102 k | 5/5 |
| BC → PPO (naive) | 6.73 | 6.70 ± 0.02 | 7.01 | 6 k | 5/5 |
| BC → PPO + KL (decay 50%) | 6.73 | 6.72 ± 0.02 | 6.91 | 6 k | 5/5 |
| BC → PPO + KL (decay 20%) | 6.73 | 6.73 ± 0.04 | 6.92 | 6 k | 5/5 |
| DAgger → PPO (naive) | 6.99 | 6.68 ± 0.01 | 6.97 | 6 k | 5/5 |
| DAgger → PPO + KL (decay 50%) | 6.99 | 6.68 ± 0.03 | 7.00 | 6 k | 5/5 |
| HuRL (from scratch) | 99.14 | 6.71 ± 0.02 | 33.86 | 191 k | 5/5 |
| BC → HuRL | 6.73 | 6.71 ± 0.03 | 7.14 | 6 k | 5/5 |
| HEPO (from scratch) | 99.14 | 6.75 ± 0.03 | 18.14 | 164 k | 5/5 |
| IQL → PPO + KL | 6.98 | 6.70 ± 0.03 | 6.99 | 6 k | 5/5 |
| IQL + HUBL → PPO + KL | 6.98 | 6.74 ± 0.03 | 6.98 | 6 k | 5/5 |

Variants that use Base stock offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its 6,400 steps in their step counts; HEPO's heuristic rollouts are half of its PPO steps. 'Beat' is measured on the 64 evaluation episodes of the learning curves.
