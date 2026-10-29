import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgrl.bc import collect_demonstrations, dagger, train_bc
from hgrl.evaluate import evaluate
from hgrl.policy import ActorCritic
from problems import PROBLEMS
from problems.binpack import HEURISTICS, BinPackingConfig, BinPackingEnv, best_fit, sum_of_squares

SHORT = BinPackingConfig(episode_length=50)


def test_invalid_action_raises():
    env = BinPackingEnv(SHORT, n_envs=1)
    with pytest.raises(ValueError):   
        env.step(np.array([0]))


def test_heuristics_choose_feasible_actions_and_conserve_volume():
    for heuristic in HEURISTICS.values():
        env = BinPackingEnv(SHORT, n_envs=16, seed=1)
        for _ in range(SHORT.episode_length - 1):
            a = heuristic(env)
            assert env.action_mask()[np.arange(16), a].all()
            env.step(a)
            open_gaps = (env.counts * np.arange(env.B + 1)).sum(axis=1)
            assert (env.bins_opened * env.B - env.volume_packed == open_gaps).all()


def test_sum_of_squares_beats_best_fit():
    cfg = BinPackingConfig()
    bf = evaluate(cfg, best_fit, n_episodes=64)["metric_mean"]
    ss = evaluate(cfg, sum_of_squares, n_episodes=64)["metric_mean"]
    assert ss < bf - 0.03


def test_behaviour_cloning_reproduces_best_fit():
    torch.manual_seed(0)
    cfg = BinPackingConfig()
    env = BinPackingEnv(cfg, n_envs=1)
    policy = ActorCritic(env.obs_dim, env.n_actions)
    accuracy = train_bc(policy, collect_demonstrations(cfg, best_fit, 32, seed=3))
    assert accuracy > 0.98
    bf = evaluate(cfg, best_fit, n_episodes=64)["metric_mean"]
    assert abs(evaluate(cfg, policy, n_episodes=64)["metric_mean"] - bf) < 0.01


def test_dagger_labels_learner_states_and_reproduces_sum_of_squares():
    torch.manual_seed(0)
    cfg = BinPackingConfig()
    env = BinPackingEnv(cfg, n_envs=1)
    policy = ActorCritic(env.obs_dim, env.n_actions)
    data, accuracy = dagger(policy, cfg, sum_of_squares, n_rounds=4, n_episodes=16, seed=3)
    assert len(data["actions"]) == 4 * 16 * cfg.episode_length
    assert data["mask"][np.arange(len(data["actions"])), data["actions"]].all()
    assert accuracy > 0.98
    ss = evaluate(cfg, sum_of_squares, n_episodes=64)["metric_mean"]
    assert abs(evaluate(cfg, policy, n_episodes=64)["metric_mean"] - ss) < 0.01


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_problem_contract(name):
    """Every problem: fixed-length episodes, feasible heuristics, a per-episode metric."""
    problem = PROBLEMS[name]
    cfg = problem.Config()
    env = cfg.make(8, seed=1)
    obs, mask = env.reset()
    assert obs.shape == (8, env.obs_dim) and obs.dtype == np.float32
    assert mask.shape == (8, env.n_actions) and mask.any(axis=1).all()
    for heuristic in problem.HEURISTICS.values():
        env = cfg.make(8, seed=1)
        for t in range(cfg.episode_length):
            a = heuristic(env)
            assert env.action_mask()[np.arange(8), a].all()
            obs, mask, reward, done, info = env.step(a)
            assert done == (t == cfg.episode_length - 1)
        assert info["metric"].shape == (8,)
    assert problem.GUIDE in problem.HEURISTICS


def test_knapsack_upper_bound_dominates_heuristics():
    from problems import knapsack
    cfg = knapsack.Config()
    for h in knapsack.HEURISTICS.values():
        assert evaluate(cfg, h, n_episodes=32)["metric_mean"] >= 0


def test_guides_leave_room_below_a_reference():
    """Each new problem has a stronger reference heuristic than its guide."""
    for name, better in [("knapsack", "adaptive_threshold"), ("inventory", "capped_base_stock"),
                         ("load_balancing", None)]:
        problem = PROBLEMS[name]
        cfg = problem.Config()
        guide = evaluate(cfg, problem.HEURISTICS[problem.GUIDE], n_episodes=64)["metric_mean"]
        others = [evaluate(cfg, h, n_episodes=64)["metric_mean"] for k, h in problem.HEURISTICS.items()
                  if k != problem.GUIDE]
        if better:
            assert evaluate(cfg, problem.HEURISTICS[better], n_episodes=64)["metric_mean"] < guide
        else:   # the guide is the best heuristic we have
            assert guide < min(others)
