import numpy as np
import torch
import torch.nn.functional as F



def collect_demonstrations(cfg, heuristic, n_episodes, seed, policy=None):
    """Label every visited state with the heuristic's action.

    The heuristic also chooses the moves, unless `policy` is given: then the
    policy acts (greedily) and the heuristic only labels, as in DAgger.
    """
    env = cfg.make(n_episodes, seed=seed)
    obs, mask = env.reset()
    data = {"obs": [], "mask": [], "actions": [], "rewards": []}
    for _ in range(cfg.episode_length):
        label = heuristic(env)
        data["obs"].append(obs)
        data["mask"].append(mask)
        data["actions"].append(label)
        action = label if policy is None else policy.act(obs, mask, greedy=True)
        obs, mask, reward, _, _ = env.step(action)
        data["rewards"].append(reward.astype(np.float32))
    return {k: np.concatenate(v) for k, v in data.items()}


def train_bc(policy, data, epochs=10, batch_size=512, lr=1e-3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    obs, mask, actions = (torch.as_tensor(data[k]) for k in ("obs", "mask", "actions"))
    opt = torch.optim.Adam(policy.actor.parameters(), lr=lr)
    correct = 0
    for _ in range(epochs):
        correct = 0
        for idx in torch.randperm(len(actions), generator=gen).split(batch_size):
            logits = policy.logits(obs[idx], mask[idx])
            loss = F.cross_entropy(logits, actions[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(-1) == actions[idx]).sum().item()
    return correct / len(actions)


def dagger(policy, cfg, heuristic, n_rounds, n_episodes, seed=0):
    """DAgger (Ross et al., 2011): round 0 clones heuristic rollouts, later rounds
    roll out the current policy, label its states with the heuristic, and retrain
    on everything collected so far. Uses n_rounds * n_episodes environment episodes.
    """
    data, accuracy = None, 0.0
    for k in range(n_rounds):
        new = collect_demonstrations(cfg, heuristic, n_episodes, seed=seed + k,
                                     policy=None if k == 0 else policy)
        data = new if data is None else {key: np.concatenate([data[key], new[key]]) for key in data}
        accuracy = train_bc(policy, data, seed=seed + k)
    return data, accuracy
