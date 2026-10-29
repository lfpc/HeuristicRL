"""Online (stochastic) knapsack: items with random weight and value arrive one by
one and must be accepted or rejected on the spot, under a capacity budget.

Metric: optimality gap 1 - value / UB, with UB the offline fractional-knapsack
optimum over the episode's items (an upper bound on any policy).
"""
from dataclasses import dataclass

import numpy as np

TITLE = "Online knapsack"
DESCRIPTION = "200 items, weight and value i.i.d. uniform on {1..10}, capacity 250"
METRIC = "Gap to offline optimum (%)"
GUIDE = "static_threshold"
PLOT = dict(scale=100, unit="%", y_min=0.0, y_max=None,
            lines=[("static_threshold", "Static threshold"), ("adaptive_threshold", "Adaptive threshold")])


@dataclass(frozen=True)
class Config:
    episode_length: int = 200
    capacity: int = 250
    max_weight: int = 10
    max_value: int = 10
    reward_scale: float = 0.02   # reward = reward_scale * value of an accepted item

    def make(self, n_envs, seed=0):
        return KnapsackEnv(self, n_envs, seed)


class KnapsackEnv:
    n_actions = 2   # 0 = reject, 1 = accept
    obs_dim = 6

    def __init__(self, cfg, n_envs, seed=0):
        self.cfg, self.n_envs, self.T = cfg, n_envs, cfg.episode_length
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        c, shape = self.cfg, (self.n_envs, self.T)
        self.w = self.rng.integers(1, c.max_weight + 1, size=shape)
        self.v = self.rng.integers(1, c.max_value + 1, size=shape)
        self.t = 0
        self.cap = np.full(self.n_envs, c.capacity)
        self.value = np.zeros(self.n_envs)
        self.ub = fractional_optimum(self.w, self.v, c.capacity)
        return self._obs(), self.action_mask()

    @property
    def item(self):
        return self.w[:, self.t], self.v[:, self.t]

    def action_mask(self):
        w, _ = self.item
        return np.stack([np.ones(self.n_envs, bool), w <= self.cap], axis=1)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if not self.action_mask()[np.arange(self.n_envs), actions].all():
            raise ValueError("invalid action: item does not fit")
        w, v = self.item
        take = actions == 1
        self.cap -= np.where(take, w, 0)
        self.value += np.where(take, v, 0)
        reward = self.cfg.reward_scale * np.where(take, v, 0).astype(np.float64)
        self.t += 1
        done = self.t == self.T
        info = {}
        if done:
            info = {"metric": 1.0 - self.value / self.ub, "value": self.value.copy()}
            obs, mask = self.reset()
        else:
            obs, mask = self._obs(), self.action_mask()
        return obs, mask, reward, done, info

    def _obs(self):
        c = self.cfg
        w, v = self.item
        left = self.T - self.t
        expected_w = left * (c.max_weight + 1) / 2   # expected weight of the items still to come
        return np.stack([w / c.max_weight, v / c.max_value, (v / w) / c.max_value,
                         self.cap / c.capacity, np.full(self.n_envs, self.t / self.T),
                         np.minimum(self.cap / expected_w, 2.0) / 2], axis=1).astype(np.float32)


def fractional_optimum(w, v, capacity):
    """Per row: best value of a fractional knapsack over all items (greedy by density)."""
    order = np.argsort(-v / w, axis=1, kind="stable")
    w, v = np.take_along_axis(w, order, 1), np.take_along_axis(v, order, 1)
    before = np.cumsum(w, axis=1) - w                     # weight packed before each item
    frac = np.clip((capacity - before) / w, 0.0, 1.0)
    return (frac * v).sum(axis=1)


def _density_table(cfg):
    """All (w, v) pairs sorted by decreasing density, with cumulative expected weight per item."""
    w, v = np.meshgrid(np.arange(1, cfg.max_weight + 1), np.arange(1, cfg.max_value + 1))
    w, v = w.ravel(), v.ravel()
    order = np.argsort(-v / w, kind="stable")
    density = (v / w)[order]
    cum_weight = np.cumsum(w[order]) / len(w)   # E[w 1{density >= d}] per arriving item
    return density, cum_weight


def _threshold(env, weight_per_item):
    """Density threshold whose expected accepted weight per item matches the budget."""
    density, cum_weight = _density_table(env.cfg)
    idx = np.clip(np.searchsorted(cum_weight, weight_per_item), 0, len(density) - 1)
    return density[idx]


def static_threshold(env):
    """Accept iff density >= tau, tau fixed so the expected weight accepted over the
    episode equals the capacity (the classic fixed-price policy): the guide."""
    w, v = env.item
    tau = _threshold(env, env.cfg.capacity / env.T)
    return ((v / w >= tau) & (w <= env.cap)).astype(np.int64)


def adaptive_threshold(env):
    """Same, but re-solved every step for the remaining capacity and items (re-optimising
    fixed-price policy): near-optimal, the reference for how much room there is."""
    w, v = env.item
    tau = _threshold(env, env.cap / (env.T - env.t))
    return ((v / w >= tau) & (w <= env.cap)).astype(np.int64)


def greedy(env):
    """Accept whatever fits."""
    return env.action_mask()[:, 1].astype(np.int64)


def random_feasible(env):
    mask = env.action_mask()
    return np.where(mask, env.rng.random(mask.shape), -1.0).argmax(axis=1)


HEURISTICS = {"adaptive_threshold": adaptive_threshold, "static_threshold": static_threshold,
              "greedy": greedy, "random": random_feasible}
