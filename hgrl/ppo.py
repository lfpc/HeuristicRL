from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from .policy import mlp



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
    anchor_coef: float = 0.0       # online CE to the heuristic's action (needs a HeuristicObs env)
    anchor_decay_frac: float = 0.0  # 0 = constant; else linear decay to 0 by this fraction (kickstarting)
    jsrl_frac: float = 0.0         # JSRL: heuristic plays the first h steps, h: T -> 0 by this fraction
    prior_decay_frac: float = 0.0  # residual policies: prior kappa -> 0 linearly by this fraction (0 = constant)
    qfilter: bool = False          # anchor only where Q(s, a_H) > Q(s, greedy action) (Q-filter, Nair et al., 2018)
    gate: bool = False             # anchor on until the greedy policy matches the heuristic on fresh episodes, then off
    gate_episodes: int = 4         # paired episodes per gate check (every eval_every iterations; counted as steps)
    eval_every: int = 10        # iterations
    seed: int = 0


def rollout(policy, env, obs, mask, cfg, h=None, lam=1.0, guide_steps=0):
    """Run `n_steps` in every env; return the flattened batch and the last obs.

    With a heuristic value `h` this is HuRL's reshaped MDP (Cheng et al., 2021):
    reward r + (1 - lam) * gamma * h(s') and discount lam * gamma.
    With `guide_steps` the heuristic acts for the first steps of each episode
    (JSRL, Uchendu et al., 2023); those steps get learn = 0 and stay out of the loss.
    """
    T, E = cfg.n_steps, cfg.n_envs
    b_obs = np.zeros((T, E, env.obs_dim), np.float32)
    b_mask = np.zeros((T, E, env.n_actions), bool)
    b_act = np.zeros((T, E), np.int64)
    b_logp = np.zeros((T, E), np.float32)
    b_val = np.zeros((T + 1, E), np.float32)
    b_rew = np.zeros((T, E), np.float32)
    b_done = np.zeros((T, E), np.float32)
    b_hact = np.zeros((T, E), np.int64)
    b_learn = np.ones((T, E), np.float32)
    for t in range(T):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs)
            dist = policy.dist(obs_t, torch.as_tensor(mask))
            action = dist.sample()
            b_logp[t] = dist.log_prob(action).numpy()
            b_val[t] = policy.value(obs_t).numpy()
        b_obs[t], b_mask[t], b_act[t] = obs, mask, action.numpy()
        if hasattr(env, "h_action"):
            b_hact[t] = env.h_action
            if env.t < guide_steps:
                b_act[t], b_learn[t] = env.h_action, 0.0
        obs, mask, b_rew[t], done, _ = env.step(b_act[t])
        b_done[t] = float(done)
        if h is not None and not done:   # h(terminal) = 0: no bins left to open
            b_rew[t] += (1 - lam) * cfg.gamma * h(obs)
    with torch.no_grad():
        b_val[T] = policy.value(torch.as_tensor(obs)).numpy()

    adv = gae(b_rew, b_val, b_done, lam * cfg.gamma, cfg.gae_lambda)
    batch = {"obs": b_obs, "mask": b_mask, "act": b_act, "logp": b_logp,
             "adv": adv, "ret": adv + b_val[:-1], "h_act": b_hact, "learn": b_learn}
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


def update(policy, opt, batch, cfg, ref_policy, beta, critic_only, anchor=0.0, qnet=None, q_opt=None):
    """A few epochs of minibatch PPO on one batch (its learn = 1 part). With `qnet`, Q(s, a) is regressed on the
    returns of the actions taken, and the anchor applies only where Q(s, a_H) > Q(s, greedy action): where the policy
    already picks a_H there is nothing to correct, and a non-strict test would lock it onto the heuristic. Returns the
    share of states where the anchor applied (None without a Q-filter)."""
    learn = batch["learn"].nonzero().squeeze(-1)
    kept = []
    for _ in range(cfg.epochs):
        for idx in learn[torch.randperm(len(learn))].split(cfg.minibatch_size):
            if len(idx) < 2:   # JSRL: the heuristic played (almost) the whole rollout
                continue
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
                if anchor > 0:   # CE to the heuristic's action in the visited state
                    logp_h = torch.log_softmax(logits, -1).gather(1, batch["h_act"][idx, None]).squeeze(1)
                    if qnet is not None:
                        with torch.no_grad():
                            q = qnet(obs).masked_fill(~mask, -1e9)
                            keep = (q.gather(1, batch["h_act"][idx, None]).squeeze(1)
                                    > q.gather(1, logits.argmax(-1, keepdim=True)).squeeze(1)).float()
                        kept.append(keep.mean().item())
                        logp_h = logp_h * keep
                    loss = loss - anchor * logp_h.mean()
            if qnet is not None:
                q_sa = qnet(obs).gather(1, batch["act"][idx, None]).squeeze(1)
                q_opt.zero_grad()
                ((q_sa - batch["ret"][idx]) ** 2).mean().backward()
                q_opt.step()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt.step()
    return float(np.mean(kept)) if kept else None


def hurl_lambda(cfg, it, n_iters):
    """lambda_n = lambda_0 + (1 - lambda_0) tanh(omega n), increasing to 1."""
    omega = 3.0 / max(cfg.hurl_anneal_frac * n_iters, 1)   # tanh(3) = 0.995
    return cfg.hurl_lambda0 + (1 - cfg.hurl_lambda0) * np.tanh(omega * it)


def train_ppo(policy, env_cfg, cfg, eval_fn, ref_policy=None, heuristic_value=None, heuristic=None):
    """Train `policy` in place. Returns the eval metric vs. PPO environment steps.

    With `cfg.gate`, every `cfg.eval_every` iterations the greedy policy and `heuristic` play the same
    `cfg.gate_episodes` fresh training episodes; after the policy's metric is at least as good twice in a row the
    anchor is switched off for good. Those episodes are counted in the steps."""
    from .evaluate import evaluate
    torch.manual_seed(cfg.seed)
    env = env_cfg.make(cfg.n_envs, seed=cfg.seed)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    steps_per_iter = cfg.n_envs * cfg.n_steps
    n_iters = cfg.total_steps // steps_per_iter

    log = {"steps": [0], "metric": [eval_fn(policy)]}
    qnet = q_opt = None
    if cfg.qfilter:
        qnet = mlp(env.obs_dim, env.n_actions)
        q_opt = torch.optim.Adam(qnet.parameters(), lr=cfg.lr)
        log["qfilter"] = [1.0]
    released, streak, extra = None, 0, 0   # performance gate: release step, checks passed in a row, gate steps
    obs, mask = env.reset()
    for it in range(n_iters):
        lam = hurl_lambda(cfg, it, n_iters) if heuristic_value is not None else 1.0
        progress = it / max(n_iters - 1, 1)
        guide = (round(env_cfg.episode_length * max(0.0, 1.0 - progress / cfg.jsrl_frac))
                 if cfg.jsrl_frac > 0 else 0)
        if cfg.prior_decay_frac > 0:   # fading residual prior, fixed within an iteration
            policy.kappa = policy.kappa0 * max(0.0, 1.0 - progress / cfg.prior_decay_frac)
        batch, obs, mask = rollout(policy, env, obs, mask, cfg, heuristic_value, lam, guide)
        beta = cfg.kl_coef * max(0.0, 1.0 - progress / cfg.kl_decay_frac) if ref_policy else 0.0
        anchor = cfg.anchor_coef * (max(0.0, 1.0 - progress / cfg.anchor_decay_frac)
                                    if cfg.anchor_decay_frac > 0 else 1.0)
        if cfg.gate and released is not None:
            anchor = 0.0
        kept = update(policy, opt, batch, cfg, ref_policy, beta, critic_only=it < cfg.critic_warmup_iters,
                      anchor=anchor, qnet=qnet, q_opt=q_opt)
        if cfg.gate and released is None and (it + 1) % cfg.eval_every == 0:
            seed = 10_000_000 + 1000 * cfg.seed + it            # fresh training episodes, never the evaluation set
            mine = evaluate(env_cfg, policy, n_episodes=cfg.gate_episodes, seed=seed)["metric_mean"]
            theirs = evaluate(env_cfg, heuristic, n_episodes=cfg.gate_episodes, seed=seed)["metric_mean"]
            extra += 2 * cfg.gate_episodes * env_cfg.episode_length
            streak = streak + 1 if mine <= theirs else 0
            if streak == 2:
                released = (it + 1) * steps_per_iter + extra
                log["released"] = released
        if (it + 1) % cfg.eval_every == 0 or it == n_iters - 1:
            log["steps"].append((it + 1) * steps_per_iter + extra)
            log["metric"].append(eval_fn(policy))
            if qnet is not None:
                log["qfilter"].append(kept if kept is not None else 1.0)
    return log
