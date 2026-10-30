"""Off-policy online RL that keeps the heuristic's data in the replay buffer.

Unlike PPO, these never throw the heuristic's transitions away: every update draws
a fixed share of its batch from them (symmetric sampling, as in RLPD, Ball et al.,
2023) and the rest from the agent's own, growing replay buffer.

- IQL, offline -> online (Kostrikov et al., 2022): the IQL of hgrl/guided.py keeps
  training, critic included, while its actor collects new data.
- DQfD (Hester et al., 2018): double DQN pretrained on the heuristic's
  transitions, plus a large-margin loss that ranks the heuristic's action first
  on them. Q is written into `policy.actor`, so the greedy policy is argmax Q.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .policy import mlp

KEYS = ("obs", "mask", "actions", "rewards", "next_obs", "next_mask", "done")


class IQL:
    """Discrete IQL; the actor (advantage-weighted regression) is `policy.actor`."""

    def __init__(self, policy, obs_dim, n_actions, gamma=0.99, lr=3e-4, expectile=0.7, beta=10.0, tau=0.005,
                 adv_norm=False):
        self.policy, self.gamma, self.expectile, self.beta, self.tau = policy, gamma, expectile, beta, tau
        self.adv_norm = adv_norm   # weights exp(beta * A / std(A)): independent of the reward scale
        self.q, self.v = mlp(obs_dim, n_actions), mlp(obs_dim, 1)
        self.q_targ = copy.deepcopy(self.q).requires_grad_(False)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.opt_pi = torch.optim.Adam(policy.actor.parameters(), lr=lr)

    def update(self, o, m, a, r, disc, no, nm, done):
        q_sa = self.q_targ(o).gather(1, a[:, None]).squeeze(1)
        diff = q_sa - self.v(o).squeeze(1)
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        self.opt_v.zero_grad()
        (weight * diff ** 2).mean().backward()
        self.opt_v.step()

        with torch.no_grad():
            target = r + disc * (1 - done) * self.v(no).squeeze(1)
        self.opt_q.zero_grad()
        F.mse_loss(self.q(o).gather(1, a[:, None]).squeeze(1), target).backward()
        self.opt_q.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
                pt.lerp_(p, self.tau)

        with torch.no_grad():
            adv = q_sa - self.v(o).squeeze(1)
            if self.adv_norm:
                adv = adv / (adv.std() + 1e-8)
            w = torch.exp(self.beta * adv).clamp(max=100.0)
        logp = torch.log_softmax(self.policy.logits(o, m), -1).gather(1, a[:, None]).squeeze(1)
        self.opt_pi.zero_grad()
        (-(w * logp).mean()).backward()
        self.opt_pi.step()

    @torch.no_grad()
    def act(self, obs, mask, rng):
        return self.policy.act(obs, mask)


class DQfD:
    """Double DQN + large-margin supervised loss on the heuristic's transitions."""

    def __init__(self, policy, gamma=0.99, lr=3e-4, tau=0.005, margin=0.1, margin_coef=1.0,
                 eps_start=0.1, eps_end=0.01):
        self.policy, self.gamma, self.tau = policy, gamma, tau
        self.margin, self.margin_coef = margin, margin_coef
        self.q_targ = copy.deepcopy(policy.actor).requires_grad_(False)
        self.opt = torch.optim.Adam(policy.actor.parameters(), lr=lr)
        self.eps, self.eps_start, self.eps_end = eps_start, eps_start, eps_end

    def update(self, o, m, a, r, disc, no, nm, done, is_demo):
        q = self.policy.logits(o, m)                     # masked Q values
        q_sa = q.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            best = self.policy.logits(no, nm).argmax(1, keepdim=True)
            target = r + disc * (1 - done) * self.q_targ(no).gather(1, best).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, target)
        if is_demo.any():   # max_a [Q(s, a) + margin * (a != a_H)] - Q(s, a_H) on heuristic samples
            bonus = self.margin * (1 - F.one_hot(a, q.shape[1]).float())
            large_margin = (q + bonus).max(1).values - q_sa
            loss = loss + self.margin_coef * (large_margin * is_demo).sum() / is_demo.sum()
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.policy.actor.parameters(), self.q_targ.parameters()):
                pt.lerp_(p, self.tau)

    @torch.no_grad()
    def act(self, obs, mask, rng):
        greedy = self.policy.act(obs, mask, greedy=True)
        explore = np.where(mask, rng.random(mask.shape), -1.0).argmax(axis=1)
        return np.where(rng.random(len(greedy)) < self.eps, explore, greedy)


class Replay:
    """FIFO buffer of transitions (numpy, fixed capacity)."""

    def __init__(self, capacity, obs_dim, n_actions):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.data = {"obs": np.zeros((capacity, obs_dim), np.float32),
                     "mask": np.zeros((capacity, n_actions), bool),
                     "actions": np.zeros(capacity, np.int64),
                     "rewards": np.zeros(capacity, np.float32),
                     "next_obs": np.zeros((capacity, obs_dim), np.float32),
                     "next_mask": np.zeros((capacity, n_actions), bool),
                     "done": np.zeros(capacity, np.float32)}

    def add(self, **batch):
        n = len(batch["actions"])
        idx = (self.pos + np.arange(n)) % self.capacity
        for k in KEYS:
            self.data[k][idx] = batch[k]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)


def _batch(sources, sizes, gen):
    """Concatenate `sizes[i]` uniform samples from each source dict -> torch tensors, plus is_demo."""
    parts = {k: [] for k in KEYS + ("disc",)}
    is_demo = []
    for (src, n_avail, disc, demo), n in zip(sources, sizes):
        idx = torch.randint(n_avail, (n,), generator=gen).numpy()
        for k in KEYS:
            parts[k].append(src[k][idx])
        parts["disc"].append(disc[idx] if isinstance(disc, np.ndarray) else np.full(n, disc, np.float32))
        is_demo.append(np.full(n, float(demo), np.float32))
    out = {k: torch.as_tensor(np.concatenate(v)) for k, v in parts.items()}
    out["is_demo"] = torch.as_tensor(np.concatenate(is_demo))
    return out


def train_offpolicy(policy, agent, env_cfg, cfg, eval_fn, demos, demo_disc, pretrain_steps=10_000,
                    batch_size=512, demo_frac=0.5, buffer_size=500_000, fade_frac=0.0):
    """Pretrain `agent` on the heuristic's transitions `demos`, then interact for
    `cfg.total_steps` env steps with `cfg.n_envs` envs, one update per vector step.
    `demo_disc` is the per-transition discount of the demos (HUBL-relabelled or gamma);
    for HUBL the demo rewards are expected already relabelled. Evaluates on the same
    cadence as PPO (every `cfg.eval_every * cfg.n_steps` vector steps). With `fade_frac` > 0 the demonstration
    share of each batch (and DQfD's margin loss) decays linearly to 0 by that fraction of the interaction."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    demo_src = (demos, len(demos["actions"]), demo_disc, True)
    for _ in range(pretrain_steps):
        b = _batch([demo_src], [batch_size], gen)
        _update(agent, b)

    env = env_cfg.make(cfg.n_envs, seed=cfg.seed)
    replay = Replay(buffer_size, env.obs_dim, env.n_actions)
    n_vec = cfg.total_steps // cfg.n_envs
    eval_every = cfg.eval_every * cfg.n_steps
    log = {"steps": [0], "metric": [eval_fn(policy)]}
    obs, mask = env.reset()
    margin0 = getattr(agent, "margin_coef", 0.0)
    for i in range(n_vec):
        fade = max(0.0, 1.0 - i / (fade_frac * n_vec)) if fade_frac > 0 else 1.0
        n_demo = int(round(demo_frac * fade * batch_size))
        if isinstance(agent, DQfD):
            agent.margin_coef = margin0 * fade
        if isinstance(agent, DQfD):   # epsilon decays linearly over the first half
            agent.eps = agent.eps_start + (agent.eps_end - agent.eps_start) * min(1.0, i / (0.5 * n_vec))
        action = agent.act(obs, mask, rng)
        next_obs, next_mask, reward, done, _ = env.step(action)
        replay.add(obs=obs, mask=mask, actions=action, rewards=reward.astype(np.float32),
                   next_obs=next_obs, next_mask=next_mask, done=np.full(cfg.n_envs, float(done), np.float32))
        obs, mask = next_obs, next_mask
        sources = [demo_src, (replay.data, replay.size, agent.gamma, False)]
        b = _batch(sources, [n_demo, batch_size - n_demo], gen) if n_demo else \
            _batch(sources[1:], [batch_size], gen)
        _update(agent, b)
        if (i + 1) % eval_every == 0 or i == n_vec - 1:
            log["steps"].append((i + 1) * cfg.n_envs)
            log["metric"].append(eval_fn(policy))
    return log


def _update(agent, b):
    args = (b["obs"], b["mask"], b["actions"], b["rewards"], b["disc"], b["next_obs"], b["next_mask"], b["done"])
    if isinstance(agent, DQfD):
        agent.update(*args, is_demo=b["is_demo"])
    else:
        agent.update(*args)
