# Overview

Final metric of each method relative to the heuristic that guides it, in % (mean ± std over seeds, 256 evaluation episodes; negative = better than the guide). Per-experiment details are in `<problem>/<guide>/summary.md`.

| Method | Online bin packing — guide: Best Fit | Online bin packing — guide: EoH-evolved | Online bin packing — guide: Sum of Squares | Lost-sales inventory — guide: Base stock | Online knapsack — guide: Static threshold | Load balancing — guide: Shortest expected delay |
|---|---|---|---|---|---|---|
| PPO from scratch | -62.9 ± 22.1 | -68.7 ± 18.6 | +52.5 ± 90.8 | -3.3 ± 0.5 | -54.8 ± 6.8 | -5.2 ± 1.4 |
| BC → PPO (naive) | -63.5 ± 26.8 | -81.9 ± 10.2 | -19.7 ± 10.5 | -3.3 ± 0.8 | -55.9 ± 4.2 | -4.4 ± 1.1 |
| BC → PPO + KL, decay 50% | -77.5 ± 8.9 | -88.5 ± 1.2 | -27.0 ± 7.1 | -3.1 ± 0.7 | -55.5 ± 3.3 | -3.3 ± 0.9 |
| BC → PPO + KL, decay 20% | -76.2 ± 7.2 | -84.6 ± 4.2 | -25.2 ± 9.0 | -3.2 ± 0.6 | -56.0 ± 5.7 | -3.9 ± 1.8 |
| DAgger → PPO (naive) | -74.2 ± 7.4 | -82.9 ± 5.5 | -17.7 ± 11.4 | -3.7 ± 0.3 | -53.2 ± 5.8 | -3.2 ± 1.0 |
| DAgger → PPO + KL, decay 50% | -80.1 ± 5.3 | -85.9 ± 3.6 | -28.1 ± 5.6 | -3.5 ± 0.4 | -49.6 ± 3.4 | -2.0 ± 0.9 |
| HuRL (from scratch) | -82.1 ± 3.4 | -84.5 ± 4.1 | +2.2 ± 30.6 | -3.4 ± 0.5 | -55.7 ± 6.6 | -5.9 ± 1.2 |
| BC → HuRL | -83.8 ± 2.7 | -85.0 ± 3.0 | -5.9 ± 13.8 | -2.8 ± 0.6 | -58.4 ± 2.7 | -4.1 ± 1.3 |
| HEPO (from scratch) | +0.0 ± 0.0 | -1.7 ± 1.5 | +1.4 ± 1.9 | -2.5 ± 0.4 | -18.2 ± 5.6 | -0.2 ± 0.2 |
| IQL → PPO + KL | -80.0 ± 2.1 | -85.1 ± 2.9 | -28.5 ± 6.9 | -3.2 ± 0.4 | -56.2 ± 6.3 | -3.5 ± 1.3 |
| IQL + HUBL → PPO + KL | -78.4 ± 2.8 | -85.9 ± 2.3 | -35.8 ± 7.6 | -3.1 ± 0.4 | -58.0 ± 5.0 | -3.2 ± 1.4 |
| Kickstarting (from scratch) | -71.1 ± 24.7 | -87.4 ± 2.6 | -39.5 ± 11.5 | -3.1 ± 0.5 | -51.7 ± 1.8 | -2.5 ± 0.9 |
| BC → PPO + heuristic anchor | -24.1 ± 20.8 | -63.0 ± 4.2 | -5.8 ± 9.0 | -0.5 ± 0.2 | -15.4 ± 4.7 | -0.4 ± 0.3 |
| BC → PPO + heuristic anchor, 4× batch | -60.8 ± 6.2 | -69.6 ± 3.4 | -13.2 ± 9.2 | -0.6 ± 0.2 | -17.5 ± 3.5 | -0.4 ± 0.2 |
| IQL + HUBL → PPO + heuristic anchor | -45.3 ± 16.9 | -66.2 ± 5.4 | -14.7 ± 6.1 | -0.0 ± 0.0 | -31.9 ± 8.0 | -0.3 ± 0.2 |
| Residual policy on heuristic | -61.4 ± 24.8 | -70.6 ± 19.5 | +8.0 ± 17.4 | -3.0 ± 0.5 | -35.4 ± 24.7 | -4.8 ± 1.7 |
| Jump-Start RL | -81.4 ± 7.1 | -87.2 ± 3.0 | +266.3 ± 236.3 | -0.4 ± 2.3 | -58.2 ± 2.6 | -4.8 ± 2.2 |
| Rollout-DAgger → PPO + KL | -79.8 ± 2.3 | -85.2 ± 3.0 | -32.5 ± 8.4 | -3.6 ± 0.4 | -59.8 ± 6.5 | -7.2 ± 0.8 |
| Rollout-DAgger → PPO + held KL | -68.9 ± 4.6 | -66.9 ± 3.5 | -16.7 ± 5.2 | -2.4 ± 0.4 | -64.7 ± 1.2 | -4.8 ± 0.7 |
| IQL offline → online | +0.1 ± 0.1 | -0.9 ± 2.0 | -0.9 ± 2.7 | +0.0 ± 0.0 | -0.3 ± 2.4 | -0.1 ± 0.2 |
| DQfD | +0.0 ± 0.0 | +1.3 ± 4.0 | +0.6 ± 1.2 | -0.0 ± 0.0 | -0.1 ± 0.2 | +0.1 ± 0.3 |
| Imitation, annealed 25% (Crafter) | -78.3 ± 8.8 | -87.1 ± 2.6 | -20.9 ± 4.3 | -3.7 ± 0.2 | -54.9 ± 6.2 | -2.9 ± 1.7 |
| Imitation, constant 0.1 (Crafter) | -33.0 ± 27.2 | -66.2 ± 4.4 | -9.4 ± 6.9 | -0.4 ± 0.2 | -16.5 ± 3.0 | -0.5 ± 0.3 |
| Residual prior β = 2 (Crafter) | -70.7 ± 13.8 | -79.1 ± 9.2 | -10.8 ± 16.7 | -2.6 ± 1.1 | -58.8 ± 2.3 | -5.6 ± 0.9 |
| Residual prior β = 4 (Crafter) | -71.2 ± 8.5 | -75.0 ± 11.3 | -5.3 ± 9.6 | -2.6 ± 0.4 | -60.9 ± 1.8 | -3.0 ± 2.3 |
| Residual prior β = 8 (Crafter) | -47.8 ± 28.5 | -66.5 ± 10.6 | +14.2 ± 11.6 | -1.6 ± 0.9 | -36.6 ± 18.8 | -2.2 ± 2.2 |
| Residual prior β = 4, fading | -74.2 ± 10.2 | -71.5 ± 11.5 | -1.4 ± 27.6 | -3.2 ± 0.4 | -58.2 ± 6.7 | -5.9 ± 1.3 |
| DQfD, fading | -65.5 ± 11.5 | -65.5 ± 13.1 | +32.8 ± 48.9 | +6.3 ± 5.5 | -61.4 ± 5.9 | -0.5 ± 3.7 |
| IQL offline → online, fading | +0.0 ± 0.1 | -7.6 ± 2.3 | +0.0 ± 0.2 | -0.0 ± 0.0 | -4.7 ± 6.1 | -0.8 ± 0.3 |
| IQL offline → online, fading, normalised AWR | -67.8 ± 10.7 | -71.2 ± 11.2 | +18.6 ± 47.9 | -3.9 ± 0.2 | -66.0 ± 2.0 | -8.5 ± 0.8 |
| IQL offline → online, normalised AWR, held | -59.2 ± 9.2 | -82.5 ± 5.4 | -43.6 ± 10.0 | -0.8 ± 0.5 | +11.1 ± 18.9 | -3.8 ± 1.4 |
| IQL offline → online, fading, normalised AWR, γ = 0.99 | -78.2 ± 17.0 | -87.7 ± 1.7 | -51.0 ± 4.0 | -3.8 ± 0.2 | +273.3 ± 14.0 | -8.9 ± 0.5 |
| BC → PPO + annealed imitation | -79.3 ± 2.4 | -86.9 ± 3.0 | -20.8 ± 6.0 | -3.6 ± 0.5 | -52.4 ± 4.8 | -2.2 ± 0.7 |
| IQL + HUBL → PPO + annealed imitation | -76.4 ± 7.4 | -86.2 ± 2.9 | -21.6 ± 7.3 | -3.1 ± 0.6 | -59.4 ± 5.6 | -3.7 ± 1.7 |
| Rollout-DAgger (no test) → PPO + KL | -87.8 ± 0.9 | -85.8 ± 2.6 | -12.8 ± 6.8 | -3.0 ± 0.4 | -62.6 ± 3.3 | -8.0 ± 0.9 |
| BC → PPO + Q-filtered imitation | -64.0 ± 7.1 | -68.2 ± 6.3 | -7.4 ± 7.1 | -0.7 ± 0.4 | -32.2 ± 8.0 | -0.5 ± 0.2 |
| IQL + HUBL → PPO + Q-filtered imitation | -66.3 ± 6.2 | -71.1 ± 5.1 | -13.9 ± 5.5 | -0.2 ± 0.4 | -39.5 ± 6.8 | -0.5 ± 0.3 |
| Performance-gated imitation | -75.7 ± 9.0 | -86.4 ± 3.6 | -21.0 ± 14.1 | -3.4 ± 0.5 | -56.0 ± 6.7 | -4.3 ± 1.5 |
| BC → PPO + performance-gated imitation | -68.3 ± 16.5 | -85.6 ± 4.7 | -14.7 ± 6.6 | -3.4 ± 0.5 | -56.8 ± 3.1 | -3.8 ± 1.2 |
| IQL + HUBL → PPO + performance-gated imitation | -72.2 ± 12.1 | -85.1 ± 2.9 | -12.2 ± 4.6 | -3.4 ± 0.5 | -60.9 ± 1.9 | -3.4 ± 2.6 |
