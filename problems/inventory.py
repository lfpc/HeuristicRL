"""Lost-sales inventory control with a lead time: each period, order up to 15 units,
which arrive 3 periods later; unmet demand is lost.

Metric: mean cost per period (holding 1 per unit left, lost-sales penalty 9 per unit).
"""
from dataclasses import dataclass

import numpy as np

TITLE = "Lost-sales inventory"
DESCRIPTION = "Poisson(5) demand, lead time 3, holding 1, lost sales 9, orders 0..15, 100 periods"
METRIC = "Mean cost per period"
GUIDE = "base_stock"
PLOT = dict(scale=1, unit="", y_min=None, y_max=None,
            lines=[("base_stock", "Base stock"), ("capped_base_stock", "Capped base stock")])


@dataclass(frozen=True)
class Config:
    episode_length: int = 100
    lead_time: int = 3
    demand_mean: float = 5.0
    holding: float = 1.0
    lost_sales: float = 9.0
    max_order: int = 15
    base_stock_level: int = 23   # best S by grid search on seed 7 (eval uses another seed)
    capped_level: int = 24       # capped base stock: order min(cap, S - position), same search
    capped_cap: int = 6
    reward_scale: float = 0.01   # reward = -reward_scale * cost

    def make(self, n_envs, seed=0):
        return InventoryEnv(self, n_envs, seed)


class InventoryEnv:
    STATE = ("demand", "on_hand", "pipeline", "cost")   # per-env arrays

    def __init__(self, cfg, n_envs, seed=0):
        self.cfg, self.n_envs, self.T = cfg, n_envs, cfg.episode_length
        self.n_actions = cfg.max_order + 1
        self.obs_dim = cfg.lead_time + 3   # on hand, pipeline, position, progress
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        c = self.cfg
        self.demand = self.rng.poisson(c.demand_mean, size=(self.n_envs, self.T))
        self.on_hand = np.full(self.n_envs, int(2 * c.demand_mean))
        self.pipeline = np.full((self.n_envs, c.lead_time), int(c.demand_mean))  # [0] arrives next
        self.t = 0
        self.cost = np.zeros(self.n_envs)
        return self._obs(), self.action_mask()

    def resample_future(self):
        """Redraw demand from this period on (unknown when the order is placed)."""
        self.demand[:, self.t:] = self.rng.poisson(self.cfg.demand_mean, size=(self.n_envs, self.T - self.t))

    @property
    def position(self):
        return self.on_hand + self.pipeline.sum(axis=1)

    def action_mask(self):
        return np.ones((self.n_envs, self.n_actions), bool)

    def step(self, actions):
        c = self.cfg
        actions = np.asarray(actions, dtype=np.int64)
        self.on_hand += self.pipeline[:, 0]
        self.pipeline = np.concatenate([self.pipeline[:, 1:], actions[:, None]], axis=1)
        d = self.demand[:, self.t]
        sold = np.minimum(self.on_hand, d)
        self.on_hand -= sold
        cost = c.holding * self.on_hand + c.lost_sales * (d - sold)
        self.cost += cost
        reward = -c.reward_scale * cost.astype(np.float64)
        self.t += 1
        done = self.t == self.T
        info = {}
        if done:
            info = {"metric": self.cost / self.T}
            obs, mask = self.reset()
        else:
            obs, mask = self._obs(), self.action_mask()
        return obs, mask, reward, done, info

    def _obs(self):
        scale = 4 * self.cfg.demand_mean
        return np.concatenate([self.on_hand[:, None] / scale, self.pipeline / scale,
                               self.position[:, None] / scale,
                               np.full((self.n_envs, 1), self.t / self.T)], axis=1).astype(np.float32)


def base_stock(env, level=None):
    """Order up to a fixed inventory position S (optimal with backorders, not with lost sales)."""
    level = env.cfg.base_stock_level if level is None else level
    return np.clip(level - env.position, 0, env.cfg.max_order)


def capped_base_stock(env, level=None, cap=None):
    """Base stock with a cap on each order (Xin, 2021): near-optimal for lost sales."""
    c = env.cfg
    level = c.capped_level if level is None else level
    cap = c.capped_cap if cap is None else cap
    return np.clip(level - env.position, 0, min(cap, c.max_order))


def constant_order(env):
    """Order the mean demand every period."""
    return np.full(env.n_envs, int(round(env.cfg.demand_mean)))


def random_order(env):
    return env.rng.integers(0, env.n_actions, size=env.n_envs)


HEURISTICS = {"base_stock": base_stock, "capped_base_stock": capped_base_stock,
              "constant": constant_order, "random": random_order}
