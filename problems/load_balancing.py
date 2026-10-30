"""Load balancing: jobs arrive at random times and are dispatched on arrival to one
of four FIFO servers with different speeds; the job size is known on arrival.

Metric: mean sojourn time (waiting + service) of the jobs in an episode.
"""
from dataclasses import dataclass

import numpy as np

TITLE = "Load balancing"
DESCRIPTION = "4 FIFO servers with speeds (0.4, 0.8, 1.2, 1.6), Exp(1) job sizes, load 0.85, 200 jobs"
METRIC = "Mean sojourn time"
GUIDE = "shortest_expected_delay"
PLOT = dict(scale=1, unit="", y_min=None, y_max=None,
            lines=[("least_work", "Least work left"), ("shortest_expected_delay", "Shortest expected delay")])


@dataclass(frozen=True)
class Config:
    episode_length: int = 200
    speeds: tuple = (0.4, 0.8, 1.2, 1.6)
    load: float = 0.85          # arrival rate * E[size] / total speed
    reward_scale: float = 0.02  # reward = -reward_scale * sojourn time of the dispatched job

    def make(self, n_envs, seed=0):
        return LoadBalancingEnv(self, n_envs, seed)


class LoadBalancingEnv:
    STATE = ("size", "gap", "work", "total")   # per-env arrays

    def __init__(self, cfg, n_envs, seed=0):
        self.cfg, self.n_envs, self.T = cfg, n_envs, cfg.episode_length
        self.speeds = np.asarray(cfg.speeds)
        self.n_actions = len(self.speeds)
        self.obs_dim = 2 * self.n_actions + 2   # per server: delay, delay incl. job; size; progress
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        shape = (self.n_envs, self.T)
        self.size = self.rng.exponential(1.0, size=shape)
        rate = self.cfg.load * self.speeds.sum()          # E[size] = 1
        self.gap = self.rng.exponential(1.0 / rate, size=shape)
        self.work = np.zeros((self.n_envs, self.n_actions))   # unfinished work per server
        self.t = 0
        self.total = np.zeros(self.n_envs)
        return self._obs(), self.action_mask()

    def resample_future(self):
        """Redraw what is not yet revealed: later job sizes and every inter-arrival time
        from the current job on (the next arrival is unknown at dispatch)."""
        rate = self.cfg.load * self.speeds.sum()
        self.size[:, self.t + 1:] = self.rng.exponential(1.0, size=(self.n_envs, self.T - self.t - 1))
        self.gap[:, self.t:] = self.rng.exponential(1.0 / rate, size=(self.n_envs, self.T - self.t))

    @property
    def job(self):
        return self.size[:, self.t]

    def action_mask(self):
        return np.ones((self.n_envs, self.n_actions), bool)

    def sojourn(self):
        """(n_envs, servers): time the current job would spend if sent to each server."""
        return (self.work + self.job[:, None]) / self.speeds

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        rows = np.arange(self.n_envs)
        time = self.sojourn()[rows, actions]
        self.total += time
        self.work[rows, actions] += self.job
        self.work = np.maximum(0.0, self.work - self.speeds * self.gap[:, self.t, None])
        reward = -self.cfg.reward_scale * time
        self.t += 1
        done = self.t == self.T
        info = {}
        if done:
            info = {"metric": self.total / self.T}
            obs, mask = self.reset()
        else:
            obs, mask = self._obs(), self.action_mask()
        return obs, mask, reward, done, info

    def _obs(self):
        delay = self.work / self.speeds
        return np.concatenate([np.log1p(delay), np.log1p(self.sojourn()), np.log1p(self.job)[:, None],
                               np.full((self.n_envs, 1), self.t / self.T)], axis=1).astype(np.float32)


def shortest_expected_delay(env):
    """Server where this job would finish first (individually optimal, myopic)."""
    return env.sojourn().argmin(axis=1)


def least_work(env):
    """Server with the least unfinished work, ignoring speeds (join-the-shortest-queue)."""
    return env.work.argmin(axis=1)


def proportional_random(env):
    """Random server, chosen with probability proportional to its speed."""
    p = env.speeds / env.speeds.sum()
    return (env.rng.random(env.n_envs)[:, None] > np.cumsum(p)[None, :]).sum(axis=1)


HEURISTICS = {"shortest_expected_delay": shortest_expected_delay, "least_work": least_work,
              "random": proportional_random}
