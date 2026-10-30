"""Online bin packing: items arrive one by one and each goes into a bin; use as few bins as possible.

Default: EoH's online bin-packing distribution (RL-AHD-OBP), integer items
round(clip(Weibull(3) * 45, 1, 100)) into bins of capacity 100. `TOY` is the small
discrete problem of the first experiments (items 3 or 4, capacity 12; problems/binpack_toy.py).
"""
from dataclasses import dataclass

import numpy as np

TITLE = "Online bin packing"
DESCRIPTION = "B = 100, items round(clip(Weibull(3) x 45, 1, 100)), 200 items"
METRIC = "Wasted bin capacity (%)"
GUIDE = "eoh_evolved"
PLOT = dict(scale=100, unit="%", y_min=0.0, y_max=None,
            lines=[("best_fit", "Best Fit"), ("sum_of_squares", "Sum of Squares"),
                   ("eoh_evolved", "EoH-evolved")])


@dataclass(frozen=True)
class BinPackingConfig:
    capacity: int = 100
    distribution: str = "weibull"        # "weibull" (shape, scale below) or "discrete" (item_sizes, item_probs)
    weibull_shape: float = 3.0
    weibull_scale: float = 45.0
    item_sizes: tuple = (3, 4)
    item_probs: tuple = (2 / 3, 1 / 3)
    episode_length: int = 200
    reward_scale: float = 0.1  # reward = -reward_scale per new bin

    def make(self, n_envs, seed=0):
        return BinPackingEnv(self, n_envs, seed)

    def sample_items(self, rng, shape):
        if self.distribution == "weibull":
            return np.round(np.clip(rng.weibull(self.weibull_shape, shape) * self.weibull_scale,
                                    1, self.capacity)).astype(np.int64)
        return rng.choice(np.asarray(self.item_sizes), size=shape, p=np.asarray(self.item_probs))


TOY = BinPackingConfig(capacity=12, distribution="discrete")


class BinPackingEnv:
    STATE = ("counts", "items", "bins_opened", "volume_packed")   # per-env arrays

    def __init__(self, cfg: BinPackingConfig, n_envs: int, seed: int = 0):
        self.cfg, self.n_envs = cfg, n_envs
        self.B, self.T = cfg.capacity, cfg.episode_length
        self.n_actions = self.B
        self.obs_dim = 2 * (self.B - 1) + 1  # gap counts, one-hot item, progress
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.counts = np.zeros((self.n_envs, self.B + 1), dtype=np.int64)
        self.items = self.cfg.sample_items(self.rng, (self.n_envs, self.T))
        self.t = 0
        self.bins_opened = np.zeros(self.n_envs, dtype=np.int64)
        self.volume_packed = np.zeros(self.n_envs, dtype=np.int64)
        return self._obs(), self.action_mask()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        rows = np.arange(self.n_envs)
        if not self.action_mask()[rows, actions].all():
            raise ValueError("invalid action: item does not fit into the chosen bin")

        size = self.current_item
        gap = actions + 1                      # gap B means "new bin"
        new_bin = gap == self.B
        self.counts[rows[~new_bin], gap[~new_bin]] -= 1   # bin leaves its gap class...
        new_gap = gap - size
        open_ = new_gap > 0
        self.counts[rows[open_], new_gap[open_]] += 1     # ...and joins the smaller one
        self.bins_opened += new_bin
        self.volume_packed += size
        reward = -self.cfg.reward_scale * new_bin.astype(np.float64)

        self.t += 1
        done = self.t == self.T
        info = {}
        if done:
            info = {"metric": self.waste(), "bins": self.bins_opened.copy()}
            obs, mask = self.reset()
        else:
            obs, mask = self._obs(), self.action_mask()
        return obs, mask, reward, done, info

    def resample_future(self):
        """Redraw the items not yet revealed (all after the current one)."""
        n = self.T - self.t - 1
        self.items[:, self.t + 1:] = self.cfg.sample_items(self.rng, (self.n_envs, n))

    @property
    def current_item(self):
        return self.items[:, self.t]

    def action_mask(self):
        """(n_envs, B) bool: bin with that gap exists (or is new) and the item fits."""
        has_bin = np.concatenate([self.counts[:, 1:self.B] > 0,
                                  np.ones((self.n_envs, 1), dtype=bool)], axis=1)
        fits = np.arange(1, self.B + 1)[None, :] >= self.current_item[:, None]
        return has_bin & fits

    def waste(self):
        """Unused fraction of opened capacity: 1 - volume / (bins * B)."""
        return 1.0 - self.volume_packed / np.maximum(self.bins_opened * self.B, 1)

    def _obs(self):
        counts = np.log1p(self.counts[:, 1:self.B]).astype(np.float32)
        item = np.zeros((self.n_envs, self.B - 1), dtype=np.float32)
        # an item of size B fits only a new bin, so it can share the last slot with B - 1
        item[np.arange(self.n_envs), np.minimum(self.current_item, self.B - 1) - 1] = 1.0
        progress = np.full((self.n_envs, 1), self.t / self.T, dtype=np.float32)
        return np.concatenate([counts, item, progress], axis=1)





def best_fit(env):
    """Tightest bin the item fits in; new bin only if none fits (the rule we clone).

    Actions are ordered by increasing gap and "new bin" is always allowed, so
    the first feasible action is exactly the best fit.
    """
    return env.action_mask().argmax(axis=1)


def sum_of_squares(env):
    """Placement minimising sum_g counts[g]^2 (Csirik et al., 2006).

    Near-optimal for discrete item distributions: used only as a reference for
    how much room there is above Best Fit.
    """
    counts, size, B = env.counts, env.current_item, env.B
    rows = np.arange(env.n_envs)
    mask = env.action_mask()
    cost = np.full((env.n_envs, B), np.inf)
    for a in range(B):
        gap = a + 1
        new_gap = gap - size
        change = np.zeros(env.n_envs)
        if gap < B:                               # bin leaves class `gap`
            change += 1 - 2 * counts[:, gap]
        joins = (new_gap > 0) & (new_gap < B)     # ...and joins class `new_gap`
        change += np.where(joins, 2 * counts[rows, np.clip(new_gap, 0, B)] + 1, 0)
        cost[:, a] = np.where(mask[:, a], change, np.inf)
    return cost.argmin(axis=1)


def eoh_score(item: int, bins: np.ndarray,
              progress: float, history_mean: float, history_std: float) -> np.ndarray:
    """EoH-evolved score, verbatim (comments trimmed) from RL-AHD-OBP,
    examples/bp_online/w10_on_half4/results/samples/samples_best.json: the best of
    its twelve LLM searches (Weibull items, capacity 100) and its distillation teacher.
    """
    epsilon = 1e-9
    tightness = 1.0 / (bins - item + epsilon)
    if history_std > 0 and progress > 0.1:
        typical_upper = history_mean + 0.5 * history_std
        best_after = bins.min() if bins.size > 0 else item
        over_penalty = np.where(
            (bins - item > typical_upper) & ((bins - item) > best_after + 0.2 * history_std),
            (bins - item - typical_upper) / (typical_upper + epsilon),
            0.0
        )
    else:
        over_penalty = np.zeros_like(bins)
    progress_influence = np.clip(progress, 0, 0.999)
    residual_mean = bins.mean() if bins.size > 0 else item
    residual_scale = (bins - residual_mean) / (bins.std() + epsilon)
    best_after = bins.min() if bins.size > 0 else item
    near_best = (bins - item) <= best_after + 0.3 * history_std
    progress_bonus = (1 - progress_influence) * residual_scale * near_best
    return tightness - 0.5 * over_penalty + 0.2 * progress_bonus


def eoh_inputs(env):
    """Per env, what EoH's evaluator (prob.online_binpack) hands `score`: the item, the
    feasible bins as counts per gap -- opened bins *plus the pool of untouched bins at
    capacity* (one per item of the episode, minus those opened) -- and progress and
    item history excluding the current item."""
    B, t, T = env.B, env.t, env.T
    item = env.items[:, t]
    counts = env.counts.copy()
    counts[:, B] = T - env.bins_opened
    counts[np.arange(B + 1)[None, :] < item[:, None]] = 0
    past = env.items[:, :t].astype(np.float64)
    history_mean = past.mean(axis=1) if t else np.zeros(env.n_envs)
    history_std = past.std(axis=1) if t > 1 else np.zeros(env.n_envs)
    return item, counts, min(1.0, t / T), history_mean, history_std


def eoh_evolved_reference(env):
    """`eoh_score` itself, once per env on the expanded bin array (slow; the spec)."""
    item, counts, progress, hm, hs = eoh_inputs(env)
    actions = np.empty(env.n_envs, dtype=np.int64)
    for i in range(env.n_envs):
        bins = np.repeat(np.arange(env.B + 1), counts[i])
        scores = eoh_score(int(item[i]), bins, progress=progress, history_mean=hm[i], history_std=hs[i])
        actions[i] = bins[np.argmax(scores)] - 1
    return actions


def eoh_evolved(env):
    """`eoh_score` vectorised over envs. Its only non-elementwise terms are the min,
    mean and std of `bins`, which follow from the counts per gap, so every bin at one
    gap gets the same score; the first best gap is the bin EoH's argmax picks. Same
    actions as `eoh_evolved_reference` (tests/test_problems.py)."""
    item, counts, progress, hm, hs = eoh_inputs(env)
    eps = 1e-9
    g = np.arange(env.B + 1, dtype=np.float64)[None, :]
    present = counts > 0
    n = counts.sum(axis=1, keepdims=True)
    mean = (counts * g).sum(axis=1, keepdims=True) / n
    std = np.sqrt((counts * (g - mean) ** 2).sum(axis=1, keepdims=True) / n)
    best_after = np.where(present, g, np.inf).min(axis=1, keepdims=True)
    item, hm, hs = item[:, None], hm[:, None], hs[:, None]
    tightness = 1.0 / (g - item + eps)
    typical_upper = hm + 0.5 * hs
    over = np.where((g - item > typical_upper) & (g - item > best_after + 0.2 * hs),
                    (g - item - typical_upper) / (typical_upper + eps), 0.0)
    over = np.where((hs > 0) & (progress > 0.1), over, 0.0)
    near_best = (g - item) <= best_after + 0.3 * hs
    bonus = (1 - np.clip(progress, 0, 0.999)) * (g - mean) / (std + eps) * near_best
    scores = np.where(present, tightness - 0.5 * over + 0.2 * bonus, -np.inf)
    return scores.argmax(axis=1) - 1


def random_feasible(env):
    """Uniformly random feasible action."""
    mask = env.action_mask()
    return np.where(mask, env.rng.random(mask.shape), -1.0).argmax(axis=1)


HEURISTICS = {"best_fit": best_fit, "sum_of_squares": sum_of_squares, "eoh_evolved": eoh_evolved,
              "random": random_feasible}

Config = BinPackingConfig
