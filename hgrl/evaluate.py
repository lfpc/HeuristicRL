"""Mean metric on a fixed set of episodes (the same random sequences for every method)."""

from .policy import ActorCritic

EVAL_SEED = 12345  # disjoint from training and demonstration seeds


def evaluate(cfg, agent, n_episodes=256, seed=EVAL_SEED):
    """`agent` is an ActorCritic (acts greedily) or a heuristic function h(env)."""
    env = cfg.make(n_episodes, seed=seed)
    obs, mask = env.reset()
    info = {}
    for _ in range(cfg.episode_length):
        if isinstance(agent, ActorCritic):
            action = agent.act(obs, mask, greedy=True)
        else:
            action = agent(env)
        obs, mask, _, _, info = env.step(action)
    # every problem reports a per-episode "metric" (lower is better), plus its own extras
    return {f"{k}_{stat}": float(getattr(v, stat)()) for k, v in info.items() for stat in ("mean", "std")}
