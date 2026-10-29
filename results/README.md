# Overview

Final metric of each method relative to the heuristic that guides it, in % (mean ± std over seeds, 256 evaluation episodes; negative = better than the guide). Per-experiment details are in `<problem>/<guide>/summary.md`.

| Method | Online bin packing — guide: Best Fit | Online bin packing — guide: Sum of Squares | Lost-sales inventory — guide: Base stock | Online knapsack — guide: Static threshold | Load balancing — guide: Shortest expected delay |
|---|---|---|---|---|---|
| PPO from scratch | -74.4 ± 8.7 | +5.3 ± 35.9 | -3.5 ± 0.2 | -60.6 ± 2.3 | -4.9 ± 1.0 |
| BC → PPO (naive) | -62.6 ± 15.3 | -17.1 ± 11.7 | -3.5 ± 0.2 | -55.5 ± 6.6 | -4.1 ± 0.9 |
| BC → PPO + KL, decay 50% | -83.2 ± 2.3 | -25.8 ± 6.9 | -3.1 ± 0.3 | -57.7 ± 1.7 | -3.3 ± 1.1 |
| BC → PPO + KL, decay 20% | -72.9 ± 11.5 | -24.6 ± 3.4 | -3.0 ± 0.6 | -54.0 ± 5.1 | -3.4 ± 1.3 |
| DAgger → PPO (naive) | -77.1 ± 10.1 | -17.6 ± 5.8 | -3.7 ± 0.1 | -50.6 ± 4.2 | -3.2 ± 0.8 |
| DAgger → PPO + KL, decay 50% | -62.1 ± 20.4 | -28.6 ± 3.6 | -3.7 ± 0.4 | -52.0 ± 4.2 | -2.3 ± 0.9 |
| HuRL (from scratch) | -82.9 ± 3.2 | -15.4 ± 16.3 | -3.2 ± 0.2 | -52.9 ± 5.6 | -6.5 ± 0.7 |
| BC → HuRL | -82.5 ± 3.2 | -12.0 ± 5.6 | -3.3 ± 0.4 | -55.7 ± 3.9 | -4.5 ± 1.1 |
| HEPO (from scratch) | +0.0 ± 0.0 | +1.7 ± 2.1 | -2.7 ± 0.4 | -17.3 ± 5.6 | -0.5 ± 0.3 |
| IQL → PPO + KL | -79.1 ± 6.2 | -36.0 ± 11.4 | -3.4 ± 0.4 | -57.8 ± 4.1 | -2.7 ± 0.8 |
| IQL + HUBL → PPO + KL | -79.4 ± 1.8 | -31.7 ± 5.7 | -2.8 ± 0.4 | -60.9 ± 2.6 | -2.9 ± 0.8 |
