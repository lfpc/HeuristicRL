"""Online bin packing: items of size 3 or 4 arrive one by one into bins of capacity 12."""
from dataclasses import dataclass

import numpy as np

TITLE = "Online bin packing"
DESCRIPTION = "B = 12, item sizes {3, 4}, p = {2/3, 1/3}, 200 items"
METRIC = "Wasted bin capacity (%)"
GUIDE = "sum_of_squares"
PLOT = dict(scale=100, unit="%", y_min=0.0, y_max=16.0,
            lines=[("best_fit", "Best Fit"), ("sum_of_squares", "Sum of Squares")])


@dataclass(frozen=True)
class BinPackingConfig:
    capacity: int = 12
    item_sizes: tuple = (3, 4)
    item_probs: tuple = (2 / 3, 1 / 3)
    episode_length: int = 200
    reward_scale: float = 0.1  # reward = -reward_scale per new bin

    def make(self, n_envs, seed=0):
        return BinPackingEnv(self, n_envs, seed)


class BinPackingEnv:

    def __init__(self, cfg: BinPackingConfig, n_envs: int, seed: int = 0):
        self.cfg, self.n_envs = cfg, n_envs
        self.B, self.T = cfg.capacity, cfg.episode_length
        self.n_actions = self.B
        self.obs_dim = 2 * (self.B - 1) + 1  # gap counts, one-hot item, progress
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.counts = np.zeros((self.n_envs, self.B + 1), dtype=np.int64)
        self.items = self.rng.choice(np.asarray(self.cfg.item_sizes), size=(self.n_envs, self.T),
                                     p=np.asarray(self.cfg.item_probs))
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
        item[np.arange(self.n_envs), self.current_item - 1] = 1.0
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


def random_feasible(env):
    """Uniformly random feasible action."""
    mask = env.action_mask()
    return np.where(mask, env.rng.random(mask.shape), -1.0).argmax(axis=1)


HEURISTICS = {"best_fit": best_fit, "sum_of_squares": sum_of_squares, "random": random_feasible}

Config = BinPackingConfig
