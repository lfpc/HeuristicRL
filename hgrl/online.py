"""Methods that query the heuristic *during* RL, not only before it.

`HeuristicObs` wraps a problem so that every observation ends with a one-hot of
the heuristic's action in that state. The policy decides what to do with it:

- 'hidden':   the networks never see it; it only serves as the target of an
              online anchor (kickstarting, Schmitt et al., 2018), so the trained
              policy runs without the heuristic;
- 'residual': residual policy learning (Silver et al., 2018) for a discrete
              heuristic: logits = f(s, a_H) + kappa * 1[a = a_H], so training
              starts at a softened heuristic and learns where to depart from it.
              The heuristic is then part of the deployed policy.
- 'prior':    the residual form of the Crafter study: the same prior on the logits,
              but the networks do not see a_H, and the actor starts small (orthogonal,
              gain 0.01) rather than at exactly zero.
"""
import numpy as np
from torch import nn

from .policy import ActorCritic, mlp


class HeuristicObs:
    """Problem config whose envs append onehot(heuristic(env)) to the observation."""

    def __init__(self, cfg, heuristic):
        self.cfg, self.heuristic = cfg, heuristic
        self.episode_length = cfg.episode_length

    def make(self, n_envs, seed=0):
        return HeuristicObsEnv(self.cfg.make(n_envs, seed), self.heuristic)


class HeuristicObsEnv:

    def __init__(self, env, heuristic):
        self.env, self.heuristic = env, heuristic
        self.n_actions = env.n_actions
        self.obs_dim = env.obs_dim + env.n_actions

    def __getattr__(self, name):   # everything else (t, rng, counts, ...) is the inner env's
        return getattr(self.env, name)

    def _augment(self, obs):
        self.h_action = np.asarray(self.heuristic(self.env), dtype=np.int64)
        onehot = np.zeros((len(obs), self.n_actions), np.float32)
        onehot[np.arange(len(obs)), self.h_action] = 1.0
        return np.concatenate([obs, onehot], axis=1)

    def reset(self):
        obs, mask = self.env.reset()
        return self._augment(obs), mask

    def step(self, actions):
        obs, mask, reward, done, info = self.env.step(actions)
        return self._augment(obs), mask, reward, done, info


def heuristic_action(obs, n_actions):
    """The heuristic's action, read back from a HeuristicObs observation."""
    return obs[..., -n_actions:].argmax(-1)


class GuidedActorCritic(ActorCritic):
    """ActorCritic on HeuristicObs observations (see the module docstring for `mode`)."""

    def __init__(self, obs_dim, n_actions, mode="hidden", kappa=3.0):
        nn.Module.__init__(self)
        self.n_actions, self.mode, self.kappa, self.kappa0 = n_actions, mode, kappa, kappa
        n_in = obs_dim - n_actions if mode in ("hidden", "prior") else obs_dim
        self.actor, self.critic = mlp(n_in, n_actions), mlp(n_in, 1)
        if mode == "residual":   # start exactly at the heuristic prior
            nn.init.zeros_(self.actor[-1].weight)
            nn.init.zeros_(self.actor[-1].bias)
        elif mode == "prior":
            nn.init.orthogonal_(self.actor[-1].weight, 0.01)
            nn.init.zeros_(self.actor[-1].bias)

    def _x(self, obs):
        return obs[..., :-self.n_actions] if self.mode in ("hidden", "prior") else obs

    def logits(self, obs, mask):
        out = self.actor(self._x(obs))
        if self.mode in ("residual", "prior"):
            out = out + self.kappa * obs[..., -self.n_actions:]
        return out.masked_fill(~mask, -1e9)

    def value(self, obs):
        return self.critic(self._x(obs)).squeeze(-1)
