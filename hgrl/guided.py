"""Heuristic-guided RL methods beyond cloning, adapted to a heuristic *policy*.

- HuRL (Cheng et al., 2021): heuristic value h(s) fitted to the heuristic's
  Monte-Carlo returns; used by `train_ppo(..., heuristic_value=h)`.
- HEPO (Lee et al., 2024): maximise (1 + alpha) J + H subject to J(pi) >= J(pi_H).
- HUBL (Geng et al., 2023): offline RL (here IQL) whose Bellman targets are
  partially replaced by Monte-Carlo heuristic values.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .policy import mlp
from .ppo import gae


# ---------------------------------------------------------------- HuRL
def returns_to_go(rewards, n_episodes, gamma=1.0):
    """Flat (T * E,) rewards in time-major order -> discounted returns-to-go."""
    r = rewards.reshape(-1, n_episodes)
    out = np.zeros_like(r)
    acc = np.zeros(n_episodes, np.float32)
    for t in reversed(range(len(r))):
        acc = r[t] + gamma * acc
        out[t] = acc
    return out.reshape(-1)


def fit_heuristic_value(data, n_episodes, epochs=50, batch_size=512, lr=1e-3, seed=0):
    """Least-squares fit of h(s) to the heuristic's Monte-Carlo returns (HuRL, Sec. 5).

    Returns a numpy function obs -> h(obs).
    """
    gen = torch.Generator().manual_seed(seed)
    obs = torch.as_tensor(data["obs"])
    target = torch.as_tensor(returns_to_go(data["rewards"], n_episodes))
    net = mlp(obs.shape[1], 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(len(obs), generator=gen).split(batch_size):
            loss = F.mse_loss(net(obs[idx]).squeeze(-1), target[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.requires_grad_(False)
    return lambda o: net(torch.as_tensor(o)).squeeze(-1).numpy()


# ---------------------------------------------------------------- HEPO
def train_hepo(policy, env_cfg, cfg, heuristic, eval_fn, h_coef=0.1, alpha0=1.0, alpha_lr=1.0):
    """HEPO with PPO (Lee et al., 2024, Alg. 1), adapted to a heuristic policy:

    - heuristic reward h(s, a) = h_coef * [a == heuristic(s)], whose optimal
      policy pi_H is the heuristic itself, so pi_H is the heuristic (not trained);
    - each iteration, half the envs are rolled out by pi and half by pi_H, on the
      same item sequences; pi is updated on both halves (Eq. 7, with
      pi_H(a|s) = 1 in the importance ratio) with advantage (1 + alpha) A_r + A_h;
    - alpha <- max(0, alpha - lr * (J(pi) - J(pi_H))) (Eq. 6), with J from the
      episodes both halves just finished (Monte-Carlo instead of Eqs. 9-10).
    """
    torch.manual_seed(cfg.seed)
    E = cfg.n_envs // 2
    envs = [env_cfg.make(E, seed=cfg.seed) for _ in range(2)]  # [pi, pi_H]
    critic_h = mlp(envs[0].obs_dim, 1)
    params = list(policy.parameters()) + list(critic_h.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    steps_per_iter = 2 * E * cfg.n_steps
    n_iters = cfg.total_steps // steps_per_iter
    alpha, T = alpha0, cfg.n_steps

    log = {"steps": [0], "metric": [eval_fn(policy)], "alpha": [alpha]}
    state = [env.reset() for env in envs]
    ep_ret = [np.zeros(E), np.zeros(E)]
    for it in range(n_iters):
        batches, finished = [], [[], []]
        for k, env in enumerate(envs):
            obs, mask = state[k]
            b = {key: [] for key in ("obs", "mask", "act", "logp", "r", "h", "done", "vr", "vh")}
            for t in range(T + 1):
                with torch.no_grad():
                    o = torch.as_tensor(obs)
                    vr, vh = policy.value(o).numpy(), critic_h(o).squeeze(-1).numpy()
                b["vr"].append(vr)
                b["vh"].append(vh)
                if t == T:
                    break
                label = heuristic(env)
                if k == 0:
                    with torch.no_grad():
                        dist = policy.dist(o, torch.as_tensor(mask))
                        act = dist.sample()
                        logp = dist.log_prob(act).numpy()
                    act = act.numpy()
                else:
                    act, logp = label, np.zeros(E, np.float32)   # deterministic pi_H
                b["obs"].append(obs)
                b["mask"].append(mask)
                b["act"].append(act)
                b["logp"].append(logp)
                b["h"].append(h_coef * (act == label))
                obs, mask, r, done, _ = env.step(act)
                b["r"].append(r)
                b["done"].append(np.full(E, float(done)))
                ep_ret[k] += r
                if done:
                    finished[k].append(ep_ret[k].mean())
                    ep_ret[k] = np.zeros(E)
            state[k] = (obs, mask)
            b = {key: np.asarray(v, np.float32) for key, v in b.items()}
            b["adv_r"] = gae(b["r"], b["vr"], b["done"], cfg.gamma, cfg.gae_lambda)
            b["adv_h"] = gae(b["h"], b["vh"], b["done"], cfg.gamma, cfg.gae_lambda)
            batches.append(b)

        # Lagrange multiplier: grows while pi is worse than the heuristic on the task.
        if finished[0] and finished[1]:
            alpha = max(0.0, alpha - alpha_lr * (np.mean(finished[0]) - np.mean(finished[1])))

        cat = lambda key: torch.as_tensor(np.concatenate([bb[key].reshape(T * E, *bb[key].shape[2:])
                                                          for bb in batches]))
        obs, mask, act, logp_old = cat("obs"), cat("mask").bool(), cat("act").long(), cat("logp")
        adv_r, adv_h = cat("adv_r"), cat("adv_h")
        ret_r = adv_r + torch.as_tensor(np.concatenate([bb["vr"][:-1].reshape(-1) for bb in batches]))
        ret_h = adv_h + torch.as_tensor(np.concatenate([bb["vh"][:-1].reshape(-1) for bb in batches]))
        adv = (1 + alpha) * adv_r + adv_h
        for _ in range(cfg.epochs):
            for idx in torch.randperm(len(act)).split(cfg.minibatch_size):
                dist = Categorical(logits=policy.logits(obs[idx], mask[idx]))
                ratio = (dist.log_prob(act[idx]) - logp_old[idx]).exp()
                a = adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
                loss = -torch.min(ratio * a, clipped * a).mean() - cfg.ent_coef * dist.entropy().mean()
                loss = loss + cfg.vf_coef * (((policy.value(obs[idx]) - ret_r[idx]) ** 2).mean()
                                             + ((critic_h(obs[idx]).squeeze(-1) - ret_h[idx]) ** 2).mean())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()

        if (it + 1) % cfg.eval_every == 0 or it == n_iters - 1:
            log["steps"].append((it + 1) * steps_per_iter)
            log["metric"].append(eval_fn(policy))
            log["alpha"].append(alpha)
    return log


# ---------------------------------------------------------------- HUBL
def collect_transitions(cfg, heuristic, n_episodes, seed, eps_max=0.5):
    """Offline dataset of mixed quality: episode i follows the heuristic but takes a
    uniformly random feasible action with probability eps_i in [0, eps_max]."""
    env = cfg.make(n_episodes, seed=seed)
    eps = np.linspace(0.0, eps_max, n_episodes)
    obs, mask = env.reset()
    data = {"obs": [], "mask": [], "actions": [], "rewards": [], "next_obs": [], "next_mask": [], "done": []}
    for _ in range(cfg.episode_length):
        rand = np.where(mask, env.rng.random(mask.shape), -1.0).argmax(axis=1)
        action = np.where(env.rng.random(n_episodes) < eps, rand, heuristic(env))
        data["obs"].append(obs)
        data["mask"].append(mask)
        data["actions"].append(action)
        obs, mask, reward, done, _ = env.step(action)
        data["rewards"].append(reward.astype(np.float32))
        data["next_obs"].append(obs)
        data["next_mask"].append(mask)
        data["done"].append(np.full(n_episodes, float(done), np.float32))
    return {k: np.concatenate(v) for k, v in data.items()}


def hubl_relabel(data, n_episodes, gamma, alpha):
    """HUBL (Geng et al., 2023, Eq. 2, rank-based lambda): r~ = r + gamma lambda h',
    gamma~ = gamma (1 - lambda), with h the Monte-Carlo return-to-go and lambda(tau)
    = alpha * (rank of the trajectory's mean h) / n. alpha = 0 recovers plain IQL."""
    h = returns_to_go(data["rewards"], n_episodes, gamma).reshape(-1, n_episodes)
    h_next = np.vstack([h[1:], np.zeros((1, n_episodes), np.float32)])   # h(terminal) = 0
    mean_h = h.mean(axis=0)
    lam = alpha * (mean_h[None, :] <= mean_h[:, None]).sum(axis=1) / n_episodes
    lam = np.broadcast_to(lam, h.shape).reshape(-1)
    return (data["rewards"] + gamma * lam * h_next.reshape(-1)).astype(np.float32), (gamma * (1 - lam)).astype(np.float32)


def train_iql(policy, data, n_episodes, gamma=0.99, hubl_alpha=0.0, steps=10_000, batch_size=512,
              lr=3e-4, expectile=0.7, beta=10.0, tau=0.005, seed=0):
    """Discrete IQL (Kostrikov et al., 2022) on `data`, optionally HUBL-relabelled.
    The actor (advantage-weighted regression) is written into `policy.actor`."""
    gen = torch.Generator().manual_seed(seed)
    rew, disc = hubl_relabel(data, n_episodes, gamma, hubl_alpha)
    obs, nobs = torch.as_tensor(data["obs"]), torch.as_tensor(data["next_obs"])
    mask, act = torch.as_tensor(data["mask"]), torch.as_tensor(data["actions"])
    rew, disc, done = torch.as_tensor(rew), torch.as_tensor(disc, dtype=torch.float32), torch.as_tensor(data["done"])
    q, v = mlp(obs.shape[1], mask.shape[1]), mlp(obs.shape[1], 1)
    q_targ = copy.deepcopy(q).requires_grad_(False)
    opt_q, opt_v = torch.optim.Adam(q.parameters(), lr=lr), torch.optim.Adam(v.parameters(), lr=lr)
    opt_pi = torch.optim.Adam(policy.actor.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(len(act), (batch_size,), generator=gen)
        o, a = obs[idx], act[idx]
        q_sa = q_targ(o).gather(1, a[:, None]).squeeze(1)
        diff = q_sa - v(o).squeeze(1)
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        opt_v.zero_grad()
        (weight * diff ** 2).mean().backward()
        opt_v.step()

        with torch.no_grad():
            target = rew[idx] + disc[idx] * (1 - done[idx]) * v(nobs[idx]).squeeze(1)
        opt_q.zero_grad()
        F.mse_loss(q(o).gather(1, a[:, None]).squeeze(1), target).backward()
        opt_q.step()
        with torch.no_grad():
            for p, pt in zip(q.parameters(), q_targ.parameters()):
                pt.lerp_(p, tau)

        with torch.no_grad():
            w = torch.exp(beta * (q_sa - v(o).squeeze(1))).clamp(max=100.0)
        logp = torch.log_softmax(policy.logits(o, mask[idx]), -1).gather(1, a[:, None]).squeeze(1)
        opt_pi.zero_grad()
        (-(w * logp).mean()).backward()
        opt_pi.step()
