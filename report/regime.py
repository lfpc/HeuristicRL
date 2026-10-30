"""When does a lasting pull towards the heuristic help? Per experiment, the budget up to which the methods that keep
the policy tied to the heuristic (constant anchor or imitation, strong residual prior, DQfD) beat PPO from scratch,
against the steps PPO from scratch needs to first beat the heuristic. Reads results/<problem>/<guide>/results.json
(learning curves on 64 episodes) and writes results/regime.{png,pdf} and results/regime.json.

  python report/regime.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {"binpack_toy/best_fit": "Bin packing, Best Fit", "binpack_toy/sum_of_squares": "Bin packing, Sum of Squares",
               "binpack_toy/eoh_evolved": "Bin packing, EoH", "knapsack/static_threshold": "Knapsack",
               "load_balancing/shortest_expected_delay": "Load balancing", "inventory/base_stock": "Inventory"}
LASTING = ["bc_anchor", "hubl_anchor", "crafter_imitate_c01", "crafter_residual_b8", "dqfd"]
FADING = ["bc_ppo_kl", "hubl_ppo_kl", "crafter_imitate", "kickstart", "rollout_dagger_kl"]
INK, MUTED, ACCENT = "#1F2937", "#6B7280", "#3353B7"
OFFSET = {"Inventory": (-8, 6, "right"), "Knapsack": (8, 4, "left"), "Load balancing": (8, -12, "left")}


def at(data, variant, budget):
    """Mean over seeds of the learning-curve value at the last evaluation within `budget` steps."""
    vals = []
    for r in data["runs"]:
        if r["variant"] == variant:
            s, m = np.array(r["steps"]), np.array(r["metric"])
            vals.append(m[s <= budget][-1] if (s <= budget).any() else m[0])
    return float(np.mean(vals))


def crossover(data, group):
    """First budget after which the group's mean no longer beats PPO from scratch (the run's end if never)."""
    grid = sorted({s for r in data["runs"] for s in r["steps"] if s > 0})
    for b in grid:
        if np.mean([at(data, v, b) for v in group]) >= at(data, "scratch", b) and b > 20_000:
            return b, False
    return grid[-1], True


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter).parse_args()
    rows = []
    for key, name in EXPERIMENTS.items():
        data = json.loads((ROOT / "results" / key / "results.json").read_text())
        cross, censored = crossover(data, LASTING)
        rows.append({"experiment": key, "name": name, "lasting_better_until": cross, "censored": censored,
                     "scratch_beats_guide": data["summary"]["scratch"]["steps_to_beat_guide"]})
    (ROOT / "results" / "regime.json").write_text(json.dumps(rows, indent=1))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = np.array([r["scratch_beats_guide"] for r in rows]) / 1e3
    y = np.array([r["lasting_better_until"] for r in rows]) / 1e3
    lim = (10, 3000)
    ax.plot(lim, lim, color=MUTED, ls=":", lw=1.2)
    ax.text(45, 36, "y = x", color=MUTED, fontsize=11)
    for r, xi, yi in zip(rows, x, y):
        ax.plot(xi, yi, "^" if r["censored"] else "o", ms=9, color=ACCENT, mec="white", mew=1.5)
        dx, dy, ha = OFFSET.get(r["name"], (8, -4, "left"))
        if r["censored"]:
            dx, dy, ha = -8, 4, "right"
        ax.annotate(r["name"], (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=10, color=INK, ha=ha)
    ax.text(12, 2300, "▲ better for the whole run (1.5M steps)", color=MUTED, fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("steps until PPO from scratch beats the guide (k)")
    ax.set_ylabel("budget up to which a lasting pull\nbeats PPO from scratch (k)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("A lasting pull towards the heuristic pays\nonly until RL alone reaches the heuristic",
                 loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "regime.png", dpi=200)
    fig.savefig(ROOT / "results" / "regime.pdf")
    for r in rows:
        print(f"{r['name']:30s} lasting better until {r['lasting_better_until'] / 1e3:6.0f}k"
              f"{' (whole run)' if r['censored'] else ''}; scratch beats the guide at {r['scratch_beats_guide'] / 1e3:.0f}k")


if __name__ == "__main__":
    main()
