"""Rollout policy improvement of the heuristic (Bertsekas, 1997; Tesauro & Galperin, 1996).

In state s, try every feasible action a, let the heuristic finish the episode on
`n_samples` freshly drawn futures (the same draws for every a), and take the
action with the best mean return. This is one step of policy iteration on the
heuristic, so in expectation it is never worse than the heuristic. It needs a
simulator: the env must be able to copy its state (`STATE`) and redraw what has
not been revealed yet (`resample_future`). Distilled into a network with DAgger,
as Expert Iteration does with its search (Anthony et al., 2017).

With few samples the argmax mostly picks noise, and those small losses add up over
an episode (bin packing with Sum of Squares: 2.2% waste -> 5.2% at 16 samples).
So an action replaces the heuristic's only when its paired gain over the heuristic
action (same futures) is `z` standard errors above zero.
"""
import copy

import numpy as np


def replicate(env, k, rng=None):
    """A copy of `env` with every env repeated k times (row i -> rows i*k .. i*k+k-1)."""
    return take(env, np.repeat(np.arange(env.n_envs), k), rng)


def take(env, rows, rng=None):
    """A copy of `env` made of the given rows (repeats allowed)."""
    sim = copy.copy(env)
    for name in env.STATE:
        setattr(sim, name, getattr(env, name)[rows])
    sim.n_envs = len(rows)
    if rng is not None:
        sim.rng = rng
    return sim


class RolloutImprovement:
    """Callable env -> improved actions; counts the simulated steps it spends."""

    def __init__(self, heuristic, n_samples=16, z=2.0, seed=0):
        self.heuristic, self.n_samples, self.z = heuristic, n_samples, z
        self.rng = np.random.default_rng(seed)
        self.sim_steps = 0

    def __call__(self, env):
        E, A, M = env.n_envs, env.n_actions, self.n_samples
        mask = env.action_mask()
        sim = replicate(env, M, self.rng)          # rows (e, m)
        sim.resample_future()                      # common random numbers across actions
        em, a = np.nonzero(np.repeat(mask, M, axis=0))   # only the feasible (e, m, a)
        sim = take(sim, em, np.random.default_rng(0))     # its end-of-episode reset must not use self.rng
        action = a
        ret = np.zeros(sim.n_envs)
        while True:
            _, _, reward, done, _ = sim.step(action)
            self.sim_steps += sim.n_envs
            ret += reward
            if done:
                break
            action = self.heuristic(sim)
        rows, base = np.arange(E), self.heuristic(env)
        full = np.zeros(E * M * A)
        full[em * A + a] = ret
        ret = full.reshape(E, M, A)                      # infeasible entries are masked out below
        gain = ret - ret[rows, :, base][:, :, None]          # paired: same future m
        se = gain.std(axis=1, ddof=1) / np.sqrt(M)
        score = np.where(mask, gain.mean(axis=1) - self.z * se, -np.inf)
        score[rows, base] = 0.0                              # the heuristic needs no evidence
        return score.argmax(axis=1)
