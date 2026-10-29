import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "scratch": dict(color="#1E9E6A", ls="-", label="PPO from scratch"),
    "bc_ppo": dict(color="#DC6027", ls="-", label="BC → PPO (naive)"),
    "bc_ppo_kl": dict(color="#3353B7", ls="-", label="BC → PPO + KL, decay 50%"),
    "bc_ppo_kl_fast": dict(color="#9B5FC0", ls="--", label="BC → PPO + KL, decay 20%"),
    "dagger_ppo": dict(color="#C7961C", ls="-.", label="DAgger → PPO (naive)"),
    "dagger_ppo_kl": dict(color="#0E9BC4", ls="-.", label="DAgger → PPO + KL, decay 50%"),
    "hurl": dict(color="#C7961C", ls="-", label="HuRL (from scratch)"),
    "bc_hurl": dict(color="#DC6027", ls="--", label="BC → HuRL"),
    "hepo": dict(color="#C2185B", ls="-", label="HEPO (from scratch)"),
    "iql_ppo_kl": dict(color="#6B7280", ls="-.", label="IQL → PPO + KL"),
    "hubl_ppo_kl": dict(color="#0E9BC4", ls="-", label="IQL + HUBL → PPO + KL"),
}
CLONING = ["scratch", "bc_ppo", "bc_ppo_kl", "bc_ppo_kl_fast", "dagger_ppo", "dagger_ppo_kl"]
GUIDED = ["scratch", "bc_ppo_kl", "hurl", "bc_hurl", "hepo", "iql_ppo_kl", "hubl_ppo_kl"]
INK, MUTED, GRID ="#1F2937", "#6B7280", "#E5E7EB"
POSTER_FIGURES = Path(__file__).resolve().parents[1] / "posters" / "figures"


def curve(runs, variant, scale):
    """Steps (millions) and mean/std of the metric over seeds."""
    rs = [r for r in runs if r["variant"] == variant]
    metric = scale * np.array([r["metric"] for r in rs])
    return np.array(rs[0]["steps"]) / 1e6, metric.mean(0), metric.std(0)


def plot_learning_curves(results_path, out_dir, pdf_dir=None, variants=CLONING, name="learning_curves"):
    from problems import PROBLEMS
    data = json.loads(Path(results_path).read_text())
    runs, base, problem = data["runs"], data["baselines"], PROBLEMS[data["problem"]]
    spec, demo = problem.PLOT, data["demo"]
    scale, unit = spec["scale"], spec["unit"]
    present = [v for v in variants if any(r["variant"] == v for r in runs)]
    curves = {v: curve(runs, v, scale) for v in present}

    # Heuristic levels to draw; the guide is always one of them.
    lines = dict(spec["lines"])
    lines.setdefault(demo, demo.replace("_", " ").capitalize())
    levels = {k: scale * base[k]["metric_mean"] for k in lines}

    # Axis range: from the problem, or around the heuristic levels and the late part of the curves.
    late = [m for x, mean, _ in curves.values() for xi, m in zip(x, mean) if xi >= 0.2 * x[-1]]
    lo, hi = min(list(levels.values()) + late), max(levels.values())
    span = max(hi - lo, 0.05 * hi)
    y_min = spec["y_min"] if spec["y_min"] is not None else max(0.0, lo - 0.5 * span)
    y_max = spec["y_max"] if spec["y_max"] is not None else hi + 2.0 * span   # room for the legend
    dy = y_max - y_min

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 15,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(10, 4.4))  # wide and flat: fits half a poster block
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    x_end = max(max(r["steps"]) for r in runs) / 1e6

    # Heuristic levels as dotted lines: the highest labelled above on the right, the rest below on the left.
    top = max(levels, key=levels.get)
    for key, y in levels.items():
        role = "guide" if key == demo else "reference"
        above = key == top
        ax.axhline(y, color=MUTED, lw=1.2, ls=":", zorder=1)
        ax.text(x_end * (0.99 if above else 0.01), y + (0.015 if above else -0.015) * dy,
                f"{lines[key]} ({role}): {y:.{1 if unit == '%' else 2}f}{unit}", color=MUTED, fontsize=13,
                va="bottom" if above else "top", ha="right" if above else "left")

    for variant, (x, mean, std) in curves.items():
        st = STYLE[variant]
        ax.fill_between(x, np.clip(mean - std, 0, None), mean + std, color=st["color"], alpha=0.10, lw=0)
        ax.plot(x, mean, color=st["color"], ls=st["ls"], lw=2, solid_capstyle="round", label=st["label"])
        ax.plot(x[-1], mean[-1], "o", ms=8, color=st["color"], mec="white", mew=2)

    # Runs from scratch start far above the axis: say so instead of squashing the plot.
    if "scratch" in curves and curves["scratch"][1][0] > y_max:
        x, mean, _ = curves["scratch"]
        ax.annotate(f"starts at {mean[0]:.{0 if unit == '%' else 1}f}{unit} ↑", (x[0], y_max),
                    xytext=(x_end * 0.04, y_max - 0.09 * dy), color=MUTED, fontsize=13,
                    arrowprops=dict(arrowstyle="-", color=STYLE["scratch"]["color"], lw=1))

    ax.set_xlim(0, x_end)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Environment steps (millions, incl. heuristic demonstrations)")
    ax.set_ylabel(problem.METRIC)
    n_seeds = len({r["seed"] for r in runs})
    ax.set_title(f"{problem.TITLE}, guided by {lines[demo]} ({n_seeds} seed{'s' * (n_seeds > 1)}, mean ± std)",
                 color=INK, loc="left", fontsize=16)
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(pdf_dir or out_dir) / f"{name}.pdf")
    fig.savefig(Path(out_dir) / f"{name}.png", dpi=200)
    plt.close(fig)


LABELS = {v: st["label"] for v, st in STYLE.items()}
ACCENT = "#3353B7"


def experiment_rows(data):
    """Per variant: final metric relative to the guide, in % (negative = better than the guide)."""
    guide = data["baselines"][data["demo"]]["metric_mean"]
    rows = {}
    for v in STYLE:
        finals = [r["final"]["metric_mean"] for r in data["runs"] if r["variant"] == v]
        if finals:
            rel = 100 * (np.array(finals) - guide) / guide
            rows[v] = (rel.mean(), rel.std(), len(finals))
    return rows


def plot_overview(results_dir):
    """One panel per experiment: every method's final metric relative to its guide heuristic,
    plus the reference heuristics as dotted lines; and a markdown table of the same numbers."""
    from problems import PROBLEMS
    paths = sorted(p for p in Path(results_dir).glob("*/*/results.json") if "quick" not in p.parts)
    exps = [json.loads(p.read_text()) for p in paths]
    order = list(STYLE)[::-1]   # first variant on top
    width = 3.2 * len(exps) + 2.6
    fig, axes = plt.subplots(1, len(exps), figsize=(width, 5.6), sharey=True)
    fig.subplots_adjust(left=2.6 / width, right=0.99, top=0.80, bottom=0.12, wspace=0.12)
    axes = np.atleast_1d(axes)
    table = {}
    for ax, data in zip(axes, exps):
        problem = PROBLEMS[data["problem"]]
        guide_key, guide = data["demo"], data["baselines"][data["demo"]]["metric_mean"]
        rows = experiment_rows(data)
        lines = dict(problem.PLOT["lines"])
        name = f"{problem.TITLE}\nguide: {lines.get(guide_key, guide_key)}"
        table[name.replace("\n", " — ")] = rows
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.axvline(0, color=INK, lw=1.2)
        refs = {}   # reference heuristics within reach of the guide (a far worse one would squash the panel)
        for key, text in lines.items():
            x = 100 * (data["baselines"][key]["metric_mean"] - guide) / guide
            if key != guide_key and -100 <= x <= 50:
                refs[key] = x
                ax.axvline(x, color=MUTED, lw=1.2, ls=":")
                ax.text(x, -0.9, f"{text} ", color=MUTED, fontsize=10, ha="right" if x > 0 else "left", va="top")
        for i, v in enumerate(order):
            if v in rows:
                m, sd, _ = rows[v]
                ax.errorbar(m, i, xerr=sd, fmt="o", ms=8, color=ACCENT, mec="white", mew=2,
                            ecolor=ACCENT, elinewidth=1.5, capsize=0)
        xs = [m - sd for m, sd, _ in rows.values()] + [m + sd for m, sd, _ in rows.values()] + [0] + list(refs.values())
        pad = 0.12 * (max(xs) - min(xs))
        ax.set_xlim(max(min(xs) - pad, -100), max(xs) + pad)
        ax.set_title(name, color=INK, loc="left", fontsize=12)
        ax.set_xlabel("final vs. guide (%)", color=INK, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LABELS[v] for v in order], fontsize=11)
    axes[0].set_ylim(-1.6, len(order) - 0.4)
    fig.suptitle("Final metric vs. the guiding heuristic (left of 0 = better than the guide; "
                 "dotted = reference heuristic)", x=0.01, ha="left", color=INK, fontsize=13)
    fig.savefig(Path(results_dir) / "overview.png", dpi=200)
    fig.savefig(Path(results_dir) / "overview.pdf")
    plt.close(fig)

    names = list(table)
    md = ["# Overview", "",
          "Final metric of each method relative to the heuristic that guides it, in % "
          "(mean ± std over seeds, 256 evaluation episodes; negative = better than the guide). "
          "Per-experiment details are in `<problem>/<guide>/summary.md`.", "",
          "| Method | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for v in STYLE:
        cells = [f"{table[n][v][0]:+.1f} ± {table[n][v][1]:.1f}" if v in table[n] else "—" for n in names]
        md.append(f"| {LABELS[v]} | " + " | ".join(cells) + " |")
    (Path(results_dir) / "README.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:] or sorted(Path(__file__).resolve().parent.glob("results/*/*/results.json")):
        out = Path(path).parent
        plot_learning_curves(path, out, variants=CLONING)
        plot_learning_curves(path, out, variants=GUIDED, name="learning_curves_guided")
    if not sys.argv[1:]:
        plot_overview(Path(__file__).resolve().parent / "results")
