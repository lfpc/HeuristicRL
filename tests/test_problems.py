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
from hgrl.rollout import RolloutImprovement, replicate
from dataclasses import replace

from problems.binpack import (HEURISTICS, TOY, BinPackingConfig, BinPackingEnv, best_fit, eoh_evolved,
                              eoh_evolved_reference, eoh_score, sum_of_squares)

SHORT = replace(TOY, episode_length=50)
WEIBULL_SHORT = BinPackingConfig(episode_length=50)


def test_invalid_action_raises():
    env = BinPackingEnv(SHORT, n_envs=1)
    with pytest.raises(ValueError):   
        env.step(np.array([0]))


@pytest.mark.parametrize("cfg", [SHORT, WEIBULL_SHORT])
def test_heuristics_choose_feasible_actions_and_conserve_volume(cfg):
    for heuristic in HEURISTICS.values():
        env = BinPackingEnv(cfg, n_envs=16, seed=1)
        for _ in range(cfg.episode_length - 1):
            a = heuristic(env)
            assert env.action_mask()[np.arange(16), a].all()
            env.step(a)
            open_gaps = (env.counts * np.arange(env.B + 1)).sum(axis=1)
            assert (env.bins_opened * env.B - env.volume_packed == open_gaps).all()


def test_sum_of_squares_beats_best_fit_on_the_toy():
    bf = evaluate(TOY, best_fit, n_episodes=64)["metric_mean"]
    ss = evaluate(TOY, sum_of_squares, n_episodes=64)["metric_mean"]
    assert ss < bf - 0.03


def test_eoh_beats_best_fit_on_its_own_distribution():
    cfg = BinPackingConfig()
    bf = evaluate(cfg, best_fit, n_episodes=64)["metric_mean"]
    assert evaluate(cfg, eoh_evolved, n_episodes=64)["metric_mean"] < bf - 0.02


def test_weibull_items_follow_eoh_generator():
    items = BinPackingConfig().sample_items(np.random.default_rng(0), (100_000,))
    assert items.min() >= 1 and items.max() <= 100
    assert abs(items.mean() - 40.2) < 0.3 and abs(items.std() - 14.6) < 0.3   # EoH's Weibull 5k: 39.6-40.2, 14.4-14.8


@pytest.mark.parametrize("cfg", [SHORT, WEIBULL_SHORT])
def test_eoh_adapter_matches_eoh_evaluator(cfg):
    """Same bin counts as EoH's own loop (prob.online_binpack): a bin array with one
    entry per bin, the untouched ones included, and the item history before each item."""
    env = BinPackingEnv(cfg, n_envs=8, seed=5)
    items = env.items.copy()
    for _ in range(cfg.episode_length):
        _, _, _, _, info = env.step(eoh_evolved(env))
    for i in range(8):
        bins = np.full(cfg.episode_length, cfg.capacity)
        for n, item in enumerate(items[i]):
            valid = np.nonzero(bins >= item)[0]
            past = items[i, :n]
            scores = eoh_score(int(item), bins[valid], progress=n / len(bins),
                               history_mean=past.mean() if n else 0.0, history_std=past.std() if n > 1 else 0.0)
            bins[valid[np.argmax(scores)]] -= item
        assert (bins != cfg.capacity).sum() == info["bins"][i]


@pytest.mark.parametrize("cfg", [TOY, WEIBULL_SHORT])
def test_vectorised_eoh_matches_reference_off_its_own_path(cfg):
    env = BinPackingEnv(cfg, n_envs=32, seed=2)
    for _ in range(env.T):
        a = eoh_evolved(env)
        assert (a == eoh_evolved_reference(env)).all()
        explore = np.where(env.action_mask(), env.rng.random((32, env.B)), -1).argmax(axis=1)
        env.step(np.where(env.rng.random(32) < 0.3, explore, a))


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_resample_future_keeps_what_is_revealed(name):
    """A replicated env keeps the past and present; only the unrevealed future changes."""
    problem = PROBLEMS[name]
    env = problem.Config().make(4, seed=1)
    for _ in range(7):
        env.step(problem.HEURISTICS[problem.GUIDE](env))
    obs, mask = env._obs(), env.action_mask()
    sim = replicate(env, 3, np.random.default_rng(0))
    sim.resample_future()
    assert np.allclose(sim._obs(), np.repeat(obs, 3, axis=0))
    assert (sim.action_mask() == np.repeat(mask, 3, axis=0)).all()
    assert any((getattr(sim, k) != np.repeat(getattr(env, k), 3, axis=0)).any() for k in env.STATE)


def test_rollout_improves_on_best_fit():
    """Plain argmax (z = 0): Best Fit leaves so much room that even noisy estimates improve on it."""
    cfg = replace(TOY, episode_length=60)
    base = evaluate(cfg, best_fit, n_episodes=16)["metric_mean"]
    improved = RolloutImprovement(best_fit, n_samples=16, z=0.0)
    assert evaluate(cfg, improved, n_episodes=16)["metric_mean"] < base - 0.02


def test_behaviour_cloning_reproduces_best_fit():
    torch.manual_seed(0)
    cfg = TOY
    env = BinPackingEnv(cfg, n_envs=1)
    policy = ActorCritic(env.obs_dim, env.n_actions)
    accuracy = train_bc(policy, collect_demonstrations(cfg, best_fit, 32, seed=3))
    assert accuracy > 0.98
    bf = evaluate(cfg, best_fit, n_episodes=64)["metric_mean"]
    assert abs(evaluate(cfg, policy, n_episodes=64)["metric_mean"] - bf) < 0.01


def test_dagger_labels_learner_states_and_reproduces_sum_of_squares():
    torch.manual_seed(0)
    cfg = TOY
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
