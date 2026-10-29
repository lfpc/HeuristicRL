from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from hgrl.evaluate import EVAL_SEED
from problems.binpack import BinPackingConfig, BinPackingEnv, best_fit, sum_of_squares

N_ITEMS = 30
COLORS = {3: "#3353B7", 4: "#DC6027"}   # item size -> colour
INK, MUTED = "#1F2937", "#6B7280"
POSTER_FIGURES = Path(__file__).resolve().parents[1] / "posters" / "figures"


def pack(heuristic, cfg, n_items):
    """Bins (lists of item sizes) built by `heuristic` on the first n_items items."""
    env = BinPackingEnv(cfg, n_envs=1, seed=EVAL_SEED)
    bins = []
    for _ in range(n_items):
        size = int(env.current_item[0])
        gap = int(heuristic(env)[0]) + 1          # gap of the chosen bin (B = new bin)
        if gap == cfg.capacity:
            bins.append([size])
        else:                                     # any open bin with that gap
            next(b for b in bins if cfg.capacity - sum(b) == gap).append(size)
        env.step([gap - 1])
    return bins


def draw(ax, bins, title, B, smallest):
    for i, b in enumerate(bins):
        level = 0
        for size in b:
            ax.add_patch(Rectangle((i + 0.1, level), 0.8, size, fc=COLORS[size], ec="white", lw=2))
            ax.text(i + 0.5, level + size / 2, str(size), ha="center", va="center", color="white",
                    fontsize=10, weight="bold")
            level += size
        gap = B - level
        if 0 < gap < smallest:                    # stranded: wasted forever
            ax.add_patch(Rectangle((i + 0.1, level), 0.8, gap, fc="white", ec=MUTED, hatch="///", lw=1))
        ax.add_patch(Rectangle((i + 0.1, 0), 0.8, B, fill=False, ec=INK, lw=1.2))   # bin outline
    ax.set_xlim(0, max(len(bins), 1))
    ax.set_ylim(0, B)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.axis("off")


def main(out=Path(__file__).resolve().parent / "results" / "binpack", pdf_dir=None):
    cfg = BinPackingConfig()
    smallest = min(cfg.item_sizes)
    results = {name: pack(h, cfg, N_ITEMS) for name, h in
               [("Best Fit", best_fit), ("Sum of Squares", sum_of_squares)]}
    n_max = max(len(b) for b in results.values())

    fig, axes = plt.subplots(1, 2, figsize=(8, 1.75), gridspec_kw={"wspace": 0.08})
    for ax, (name, bins) in zip(axes, results.items()):
        stranded = sum(1 for b in bins if 0 < cfg.capacity - sum(b) < smallest)
        draw(ax, bins, f"{name}: {len(bins)} bins, {stranded} stranded gaps", cfg.capacity, smallest)
        ax.set_xlim(0, n_max)                     # same bin width in both panels
    fig.legend(handles=[Patch(fc=COLORS[3], label="item of size 3"),
                        Patch(fc=COLORS[4], label="item of size 4"),
                        Patch(fc="white", ec=MUTED, hatch="///", label="stranded gap (< 3)")],
               loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.24)
    fig.savefig(Path(pdf_dir or out) / "packing_example.pdf")
    fig.savefig(out / "packing_example.png", dpi=200)
    print({k: [sum(b) for b in v] for k, v in results.items()})


if __name__ == "__main__":
    main()
