from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical



@dataclass
class PPOConfig:
    total_steps: int = 1_500_000
    n_envs: int = 16            # parallel episodes
    n_steps: int = 128          # rollout length per env and iteration
    epochs: int = 4
    minibatch_size: int = 512
    lr: float = 3e-4
    gamma: float = 1.0          # finite horizon: total number of bins
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    critic_warmup_iters: int = 0   # fixes for warm-starting (off = naive)
    kl_coef: float = 0.0
    kl_decay_frac: float = 0.5
    hurl_lambda0: float = 1.0      # HuRL mixing lambda at the start (1 = off)
    hurl_anneal_frac: float = 0.5  # lambda reaches ~1 after this fraction of training
    eval_every: int = 10        # iterations
    seed: int = 0


def rollout(policy, env, obs, mask, cfg, h=None, lam=1.0):
    """Run `n_steps` in every env; return the flattened batch and the last obs.

    With a heuristic value `h` this is HuRL's reshaped MDP (Cheng et al., 2021):
    reward r + (1 - lam) * gamma * h(s') and discount lam * gamma.
    """
    T, E = cfg.n_steps, cfg.n_envs
    b_obs = np.zeros((T, E, env.obs_dim), np.float32)
    b_mask = np.zeros((T, E, env.n_actions), bool)
    b_act = np.zeros((T, E), np.int64)
    b_logp = np.zeros((T, E), np.float32)
    b_val = np.zeros((T + 1, E), np.float32)
    b_rew = np.zeros((T, E), np.float32)
    b_done = np.zeros((T, E), np.float32)
    for t in range(T):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs)
            dist = policy.dist(obs_t, torch.as_tensor(mask))
            action = dist.sample()
            b_logp[t] = dist.log_prob(action).numpy()
            b_val[t] = policy.value(obs_t).numpy()
        b_obs[t], b_mask[t], b_act[t] = obs, mask, action.numpy()
        obs, mask, b_rew[t], done, _ = env.step(b_act[t])
        b_done[t] = float(done)
        if h is not None and not done:   # h(terminal) = 0: no bins left to open
            b_rew[t] += (1 - lam) * cfg.gamma * h(obs)
    with torch.no_grad():
        b_val[T] = policy.value(torch.as_tensor(obs)).numpy()

    adv = gae(b_rew, b_val, b_done, lam * cfg.gamma, cfg.gae_lambda)
    batch = {"obs": b_obs, "mask": b_mask, "act": b_act, "logp": b_logp,
             "adv": adv, "ret": adv + b_val[:-1]}
    batch = {k: torch.as_tensor(v.reshape(T * E, *v.shape[2:])) for k, v in batch.items()}
    return batch, obs, mask


def gae(rew, val, done, gamma, lam):
    """Generalised advantage estimation, backwards in time."""
    adv = np.zeros_like(rew)
    last = np.zeros(rew.shape[1], np.float32)
    for t in reversed(range(len(rew))):
        not_done = 1.0 - done[t]
        delta = rew[t] + gamma * val[t + 1] * not_done - val[t]
        last = delta + gamma * lam * not_done * last
        adv[t] = last
    return adv


def masked_kl(logits_p, logits_q, mask):
    """KL(p || q) per state, over feasible actions only."""
    logp, logq = torch.log_softmax(logits_p, -1), torch.log_softmax(logits_q, -1)
    return torch.where(mask, logp.exp() * (logp - logq), torch.zeros_like(logp)).sum(-1)


def update(policy, opt, batch, cfg, ref_policy, beta, critic_only):
    """A few epochs of minibatch PPO on one batch."""
    n = len(batch["act"])
    for _ in range(cfg.epochs):
        for idx in torch.randperm(n).split(cfg.minibatch_size):
            obs, mask = batch["obs"][idx], batch["mask"][idx]
            loss = cfg.vf_coef * ((policy.value(obs) - batch["ret"][idx]) ** 2).mean()
            if not critic_only:
                logits = policy.logits(obs, mask)
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(batch["act"][idx]) - batch["logp"][idx]).exp()
                a = batch["adv"][idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
                loss = loss - torch.min(ratio * a, clipped * a).mean()
                loss = loss - cfg.ent_coef * dist.entropy().mean()
                if beta > 0:
                    with torch.no_grad():
                        ref_logits = ref_policy.logits(obs, mask)
                    loss = loss + beta * masked_kl(logits, ref_logits, mask).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt.step()


def hurl_lambda(cfg, it, n_iters):
    """lambda_n = lambda_0 + (1 - lambda_0) tanh(omega n), increasing to 1."""
    omega = 3.0 / max(cfg.hurl_anneal_frac * n_iters, 1)   # tanh(3) = 0.995
    return cfg.hurl_lambda0 + (1 - cfg.hurl_lambda0) * np.tanh(omega * it)


def train_ppo(policy, env_cfg, cfg, eval_fn, ref_policy=None, heuristic_value=None):
    """Train `policy` in place. Returns the eval metric vs. PPO environment steps."""
    torch.manual_seed(cfg.seed)
    env = env_cfg.make(cfg.n_envs, seed=cfg.seed)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    steps_per_iter = cfg.n_envs * cfg.n_steps
    n_iters = cfg.total_steps // steps_per_iter

    log = {"steps": [0], "metric": [eval_fn(policy)]}
    obs, mask = env.reset()
    for it in range(n_iters):
        lam = hurl_lambda(cfg, it, n_iters) if heuristic_value is not None else 1.0
        batch, obs, mask = rollout(policy, env, obs, mask, cfg, heuristic_value, lam)
        progress = it / max(n_iters - 1, 1)
        beta = cfg.kl_coef * max(0.0, 1.0 - progress / cfg.kl_decay_frac) if ref_policy else 0.0
        update(policy, opt, batch, cfg, ref_policy, beta, critic_only=it < cfg.critic_warmup_iters)
        if (it + 1) % cfg.eval_every == 0 or it == n_iters - 1:
            log["steps"].append((it + 1) * steps_per_iter)
            log["metric"].append(eval_fn(policy))
    return log
