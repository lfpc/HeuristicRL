"""The small discrete bin-packing problem of the first experiments: items of size 3 or 4
(p = 2/3, 1/3) into bins of capacity 12. Same env and heuristics as problems/binpack.py."""
from .binpack import HEURISTICS, TOY, BinPackingEnv, eoh_evolved, sum_of_squares  # noqa: F401
from .binpack import PLOT as _PLOT

TITLE = "Online bin packing (toy)"
DESCRIPTION = "B = 12, item sizes {3, 4}, p = {2/3, 1/3}, 200 items"
METRIC = "Wasted bin capacity (%)"
GUIDE = "sum_of_squares"
PLOT = dict(_PLOT, y_max=19.0)


def Config():
    return TOY
