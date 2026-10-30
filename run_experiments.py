"""Heuristic-guided RL on toy combinatorial problems.

Trains every variant (PPO from scratch, BC/DAgger warm starts, HuRL, HEPO,
IQL/HUBL) for several seeds on one problem, guided by one of its heuristics,
and writes results/<problem>/<heuristic>/{results.json, summary.md, *.png}.

  python run_experiments.py --problem knapsack            # problem's default guide
  python run_experiments.py --problem binpack --demo best_fit    # Weibull items, capacity 100 (EoH's)
  python run_experiments.py --problem binpack_toy             # items 3 or 4, capacity 12
  python run_experiments.py --problem binpack --no-train  # re-summarise / re-plot only
"""
import argparse
import copy
import json
import os
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from hgrl.bc import collect_demonstrations, dagger, train_bc
from hgrl.evaluate import evaluate
from hgrl.guided import collect_transitions, fit_heuristic_value, train_hepo, train_iql
from hgrl.offpolicy import DQfD, IQL, train_offpolicy
from hgrl.online import GuidedActorCritic, HeuristicObs
from hgrl.rollout import RolloutImprovement
from hgrl.policy import ActorCritic
from hgrl.ppo import PPOConfig, train_ppo
from plot_results import CLONING, CRAFTER, FADE as FADE_FIGURE, GUIDED, ONLINE, POSTER_FIGURES, RELEASE, plot_learning_curves
from problems import PROBLEMS

HERE = Path(__file__).resolve().parent
N_DEMO_EPISODES = 64
DAGGER_ROUNDS = 4   # same total demo budget as BC: 4 rounds x 16 episodes
KL = dict(critic_warmup_iters=10, kl_coef=1.0, kl_decay_frac=0.5)
HURL = dict(hurl_lambda0=0.9)   # HuRL horizon starts at ~1 / (1 - 0.9) = 10 steps
HUBL_ALPHA = 0.1                 # HUBL's recommended blending strength
# Methods that query the heuristic during RL (hgrl/online.py)
ANCHOR = dict(critic_warmup_iters=10, anchor_coef=0.1)          # held, as in RL-AHD-OBP's fine-tuning
KICKSTART = dict(anchor_coef=1.0, anchor_decay_frac=0.5)        # Schmitt et al. (2018)
BIG = dict(n_envs=64)                                           # 4x the states per update
RESIDUAL_KAPPA = 3.0                                            # heuristic prior logit
ROLLOUT_SAMPLES = 32                                            # futures per action in rollout labels
HELD_KL = dict(critic_warmup_iters=10, kl_coef=0.1, kl_decay_frac=1e9)   # constant KL to the clone
SLOW = ("iql_online", "dqfd", "rollout", "rollout_z0", "hepo")   # warm starts / methods that take far longer than PPO
VARIANTS = {   # (warm start / method, PPO options)
    "scratch": (None, {}),
    "bc_ppo": ("bc", {}),
    "bc_ppo_kl": ("bc", KL),
    "bc_ppo_kl_fast": ("bc", dict(KL, kl_decay_frac=0.2)),
    "dagger_ppo": ("dagger", {}),
    "dagger_ppo_kl": ("dagger", KL),
    "hurl": ("hurl", HURL),          # HuRL from scratch: heuristic only via h(s)
    "bc_hurl": ("bc_hurl", HURL),    # HuRL + BC warm start, as in the HuRL paper
    "hepo": ("hepo", {}),            # HEPO from scratch: heuristic only via h(s, a) and pi_H
    "iql_ppo_kl": ("iql", KL),       # offline IQL on noisy heuristic data -> PPO + KL
    "hubl_ppo_kl": ("hubl", KL),     # same, with HUBL-relabelled Bellman targets
    "kickstart": (None, KICKSTART),               # PPO + decaying CE to the heuristic on visited states
    "bc_anchor": ("bc", ANCHOR),                  # BC -> PPO + constant CE to the heuristic
    "bc_anchor_big": ("bc", dict(ANCHOR, **BIG)), # same, 4x batch
    "hubl_anchor": ("hubl", ANCHOR),              # IQL + HUBL -> PPO + constant CE to the heuristic
    "residual": ("residual", dict(critic_warmup_iters=10)),   # residual policy on the heuristic
    "jsrl": (None, dict(jsrl_frac=0.5)),          # Jump-Start RL, heuristic prefix shrinking to 0
    "rollout_dagger_kl": ("rollout", KL),         # DAgger on rollout-improved labels -> PPO + KL
    "rollout_dagger_held": ("rollout", HELD_KL),  # same, KL to the clone held at 0.1
    "iql_online": ("iql_online", {}),             # IQL offline -> online, heuristic data kept in the buffer
    "dqfd": ("dqfd", {}),                         # DQfD: double DQN + margin loss on heuristic transitions
    # The arms of the Crafter study (docs/comparison_crafter.md), as there: from scratch, no critic warm-up
    "crafter_imitate": (None, dict(anchor_coef=1.0, anchor_decay_frac=0.25)),  # imitation annealed by 25%
    "crafter_imitate_c01": (None, dict(anchor_coef=0.1)),                    # constant imitation 0.1
    "crafter_residual_b2": ("prior", {}),        # prior beta * onehot(a_H) on the logits; nets do not see a_H
    "crafter_residual_b4": ("prior", {}),
    "crafter_residual_b8": ("prior", {}),
}
# Does the heuristic's pull have to fade? The methods that held it, with the pull fading instead
FADE = 0.5
VARIANTS.update({
    "residual_fade": ("prior", dict(prior_decay_frac=FADE)),       # Crafter residual beta 4 -> 0 by 50%
    "dqfd_fade": ("dqfd", {}),                                     # demo share 0.5 -> 0 and margin -> 0 by 50%
    "iql_online_fade": ("iql_online", {}),
    "bc_imitate": ("bc", dict(critic_warmup_iters=10, anchor_coef=1.0, anchor_decay_frac=0.25)),
    "hubl_imitate": ("hubl", dict(critic_warmup_iters=10, anchor_coef=1.0, anchor_decay_frac=0.25)),
    "iql_online_norm": ("iql_online", {}),                          # fading, AWR weights on normalised advantages
    "iql_online_norm_held": ("iql_online", {}),                     # same, heuristic data held at half of each batch
    "iql_online_norm_g99": ("iql_online", {}),                      # fading, normalised, discount 0.99 instead of 1
    "rollout_dagger_kl_z0": ("rollout_z0", KL),                    # rollout labels without the significance test
    # the pull fading by itself, state by state: imitation only where a critic prefers H's action
    "bc_qfilter": ("bc", dict(critic_warmup_iters=10, anchor_coef=1.0, qfilter=True)),
    "hubl_qfilter": ("hubl", dict(critic_warmup_iters=10, anchor_coef=1.0, qfilter=True)),
    # the pull released once the greedy policy matches the heuristic twice in a row on fresh paired episodes
    "gated": (None, dict(anchor_coef=1.0, gate=True)),
    "bc_gated": ("bc", dict(critic_warmup_iters=10, anchor_coef=1.0, gate=True)),
    "hubl_gated": ("hubl", dict(critic_warmup_iters=10, anchor_coef=1.0, gate=True)),
})
CRAFTER_BETA = {"crafter_residual_b2": 2.0, "crafter_residual_b4": 4.0, "crafter_residual_b8": 8.0,
                "residual_fade": 4.0}
ONLINE = {"kickstart", "bc_anchor", "bc_anchor_big", "hubl_anchor", "residual", "jsrl", "crafter_imitate",
          "crafter_imitate_c01", "crafter_residual_b2", "crafter_residual_b4", "crafter_residual_b8",
          "residual_fade", "bc_imitate", "hubl_imitate", "bc_qfilter", "hubl_qfilter", "gated", "bc_gated",
          "hubl_gated"}
LABELS = {
    "scratch": "PPO from scratch",
    "bc_ppo": "BC → PPO (naive)",
    "bc_ppo_kl": "BC → PPO + KL (decay 50%)",
    "bc_ppo_kl_fast": "BC → PPO + KL (decay 20%)",
    "dagger_ppo": "DAgger → PPO (naive)",
    "dagger_ppo_kl": "DAgger → PPO + KL (decay 50%)",
    "hurl": "HuRL (from scratch)",
    "bc_hurl": "BC → HuRL",
    "hepo": "HEPO (from scratch)",
    "iql_ppo_kl": "IQL → PPO + KL",
    "hubl_ppo_kl": "IQL + HUBL → PPO + KL",
    "kickstart": "Kickstarting (from scratch)",
    "bc_anchor": "BC → PPO + heuristic anchor",
    "bc_anchor_big": "BC → PPO + heuristic anchor, 4× batch",
    "hubl_anchor": "IQL + HUBL → PPO + heuristic anchor",
    "residual": "Residual policy on heuristic",
    "jsrl": "Jump-Start RL",
    "rollout_dagger_kl": "Rollout-DAgger → PPO + KL",
    "rollout_dagger_held": "Rollout-DAgger → PPO + held KL",
    "iql_online": "IQL offline → online",
    "dqfd": "DQfD",
    "crafter_imitate": "Imitation, annealed by 25% (Crafter)",
    "crafter_imitate_c01": "Imitation, constant 0.1 (Crafter)",
    "crafter_residual_b2": "Residual prior β = 2 (Crafter)",
    "crafter_residual_b4": "Residual prior β = 4 (Crafter)",
    "crafter_residual_b8": "Residual prior β = 8 (Crafter)",
    "residual_fade": "Residual prior β = 4, fading",
    "dqfd_fade": "DQfD, fading",
    "iql_online_fade": "IQL offline → online, fading",
    "iql_online_norm": "IQL offline → online, fading, normalised AWR",
    "iql_online_norm_held": "IQL offline → online, normalised AWR, held",
    "iql_online_norm_g99": "IQL offline → online, fading, normalised AWR, γ = 0.99",
    "bc_imitate": "BC → PPO + annealed imitation",
    "hubl_imitate": "IQL + HUBL → PPO + annealed imitation",
    "rollout_dagger_kl_z0": "Rollout-DAgger (no test) → PPO + KL",
    "bc_qfilter": "BC → PPO + Q-filtered imitation",
    "hubl_qfilter": "IQL + HUBL → PPO + Q-filtered imitation",
    "gated": "Performance-gated imitation (from scratch)",
    "bc_gated": "BC → PPO + performance-gated imitation",
    "hubl_gated": "IQL + HUBL → PPO + performance-gated imitation",
}


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def heuristic_name(problem, key):
    return dict(problem.PLOT["lines"]).get(key, key.replace("_", " ").capitalize())


def run_one(job):
    """Train one (variant, seed). Runs in a worker process."""
    problem_name, variant, seed, total_steps, demo = job
    problem = PROBLEMS[problem_name]
    warm_start, ppo_opts = VARIANTS[variant]
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    heuristic = problem.HEURISTICS[demo]
    env_cfg = problem.Config()
    if variant in ONLINE:   # observations carry the heuristic's action; the policy decides whether to look
        env_cfg = HeuristicObs(env_cfg, heuristic)
        heuristic = lambda env: env.h_action
    probe = env_cfg.make(1)
    probe.reset()
    if variant in ONLINE:
        mode = {"residual": "residual", "prior": "prior"}.get(warm_start, "hidden")
        policy = GuidedActorCritic(probe.obs_dim, probe.n_actions, mode=mode,
                                   kappa=CRAFTER_BETA.get(variant, RESIDUAL_KAPPA))
    else:
        policy = ActorCritic(probe.obs_dim, probe.n_actions)

    demos, ref_policy, bc_acc, h, sim_steps = None, None, None, None, 0
    # Step 1: use the heuristic offline (demo seeds are disjoint from the eval seed).
    if warm_start in ("bc", "hurl", "bc_hurl"):
        demos = collect_demonstrations(env_cfg, heuristic, N_DEMO_EPISODES, seed=1000 + seed)
    if warm_start in ("bc", "bc_hurl"):
        bc_acc = train_bc(policy, demos, seed=seed)
    elif warm_start == "dagger":
        demos, bc_acc = dagger(policy, env_cfg, heuristic, DAGGER_ROUNDS,
                               N_DEMO_EPISODES // DAGGER_ROUNDS, seed=2000 + 10 * seed)
    elif warm_start in ("rollout", "rollout_z0"):   # DAgger, labelled by the rollout-improved heuristic
        labeler = RolloutImprovement(heuristic, ROLLOUT_SAMPLES, z=0.0 if warm_start == "rollout_z0" else 2.0,
                                     seed=3000 + seed)
        demos, bc_acc = dagger(policy, env_cfg, labeler, DAGGER_ROUNDS,
                               N_DEMO_EPISODES // DAGGER_ROUNDS, seed=2000 + 10 * seed)
        sim_steps = labeler.sim_steps
    elif warm_start in ("iql_online", "dqfd"):   # off-policy: these transitions stay in every batch
        demos = collect_transitions(env_cfg, heuristic, N_DEMO_EPISODES, seed=1000 + seed,
                                    eps_max=0.5 if warm_start == "iql_online" else 0.0)
    elif warm_start in ("iql", "hubl"):
        demos = collect_transitions(env_cfg, heuristic, N_DEMO_EPISODES, seed=1000 + seed)
        train_iql(policy, demos, N_DEMO_EPISODES, hubl_alpha=HUBL_ALPHA if warm_start == "hubl" else 0.0,
                  seed=seed)
    if warm_start in ("hurl", "bc_hurl"):   # same demos give the heuristic value h(s)
        h = fit_heuristic_value(demos, N_DEMO_EPISODES, seed=seed)
    if warm_start in ("bc", "dagger", "bc_hurl", "iql", "hubl", "rollout", "rollout_z0") \
            and not ppo_opts.get("anchor_coef"):
        ref_policy = copy.deepcopy(policy).requires_grad_(False)  # frozen warm-start policy
    demo_steps = 0 if demos is None else len(demos["actions"])   # counted as environment interactions

    # Step 2: RL, evaluating on 64 fixed episodes along the way.
    cfg = PPOConfig(total_steps=total_steps, seed=seed, **ppo_opts)
    eval_fn = lambda p: evaluate(env_cfg, p, n_episodes=64)["metric_mean"]
    if warm_start == "hepo":   # heuristic rollouts are part of its PPO steps
        log = train_hepo(policy, env_cfg, cfg, heuristic, eval_fn)
    elif warm_start in ("iql_online", "dqfd"):
        # gamma 1 (0.99 for the _g99 test), as PPO: the task is a finite, undiscounted horizon and the observation includes
        # progress. beta 3, not offline IQL's 10: on knapsack, exp(10 A) on the noisy advantages of fresh
        # online data collapsed the actor (gap 4% -> 40%); at 3 it stays stable.
        agent = (IQL(policy, probe.obs_dim, probe.n_actions, gamma=0.99 if variant.endswith("_g99") else 1.0, beta=3.0,
                     adv_norm=variant.startswith("iql_online_norm")) if warm_start == "iql_online"
                 else DQfD(policy, gamma=1.0))
        log = train_offpolicy(policy, agent, env_cfg, cfg, eval_fn, demos,
                              np.full(len(demos["actions"]), agent.gamma, np.float32),
                              fade_frac=FADE if variant.endswith(("_fade", "_norm", "_g99")) else 0.0)
    else:
        log = train_ppo(policy, env_cfg, cfg, ref_policy=ref_policy, eval_fn=eval_fn, heuristic_value=h,
                        heuristic=heuristic)
    return {"variant": variant, "seed": seed, "demo": demo, "bc_accuracy": bc_acc, "alpha": log.get("alpha"), "sim_steps": sim_steps, "qfilter": log.get("qfilter"), "released": log.get("released"),
            "steps": [s + demo_steps for s in log["steps"]], "metric": log["metric"],
            "final": evaluate(env_cfg, policy), "ppo_config": asdict(cfg)}


def summarise(runs, problem, demo):
    """Per variant: final metric, worst metric during training, steps to beat the guide.

    "Beat the guide" compares against the heuristic on the same 64 evaluation
    episodes the learning curves use.
    """
    target = evaluate(problem.Config(), problem.HEURISTICS[demo], n_episodes=64)["metric_mean"]
    rows = {}
    for v in VARIANTS:
        rs = [r for r in runs if r["variant"] == v]
        if not rs:
            continue
        final = np.array([r["final"]["metric_mean"] for r in rs])
        beat = [next((s for s, m in zip(r["steps"], r["metric"]) if m < target), np.nan) for r in rs]
        rows[v] = {"final_mean": final.mean(), "final_std": final.std(),
                   "initial": np.mean([r["metric"][0] for r in rs]),
                   "worst_during_training": np.mean([max(r["metric"][1:]) for r in rs]),
                   "steps_to_beat_guide": np.nanmedian(beat) if not np.isnan(beat).all() else None,
                   "seeds_beating_guide": int(np.sum(~np.isnan(beat))), "n_seeds": len(rs)}
    return rows


def write_summary(rows, baselines, path, problem, demo):
    s, unit = problem.PLOT["scale"], problem.PLOT["unit"]
    fmt = lambda x: f"{s * x:.2f}{unit}"
    guide = heuristic_name(problem, demo)
    lines = [f"# {problem.TITLE}: guided by {guide}", "",
             f"{problem.METRIC} (lower is better), greedy policies on 256 fixed episodes",
             f"({problem.DESCRIPTION}).", "",
             "| Heuristic | Metric |", "|---|---|"]
    lines += [f"| {heuristic_name(problem, k)}{' (guide)' if k == demo else ''} | "
              f"{fmt(b['metric_mean'])} ± {s * b['metric_std']:.2f} |" for k, b in baselines.items()]
    lines += ["", f"| Variant | Initial | Final | Worst during training | Steps to beat {guide} (median) | Seeds beating {guide} |",
              "|---|---|---|---|---|---|"]
    for v, r in rows.items():
        steps = "never" if r["steps_to_beat_guide"] is None else f"{r['steps_to_beat_guide'] / 1e3:.0f} k"
        lines.append(f"| {LABELS[v]} | {fmt(r['initial'])} | {fmt(r['final_mean'])} ± {s * r['final_std']:.2f} "
                     f"| {fmt(r['worst_during_training'])} | {steps} | {r['seeds_beating_guide']}/{r['n_seeds']} |")
    T = problem.Config().episode_length
    lines += ["", f"Variants that use {guide} offline (BC, DAgger, HuRL's h, IQL/HUBL data) include its "
                  f"{N_DEMO_EPISODES * T:,} steps in their step counts; "
                  "HEPO's heuristic rollouts are half of its PPO steps. "
                  f"'Beat' is measured on the 64 evaluation episodes of the learning curves.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    parser.add_argument("--problem", choices=list(PROBLEMS), default="binpack")
    parser.add_argument("--demo", help="heuristic that guides the methods (default: the problem's GUIDE)")
    parser.add_argument("--quick", action="store_true", help="1 seed, 200k steps, written to results/quick/")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1_500_000, help="PPO steps per run")
    parser.add_argument("--reuse", action="store_true",
                        help="keep runs already in the output results.json (same seeds, steps); train only missing ones")
    parser.add_argument("--no-train", action="store_true", help="only re-summarise and re-plot the saved runs")
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS),
                        help="train only these (others are kept from results.json with --reuse)")
    args = parser.parse_args()
    problem = PROBLEMS[args.problem]
    demo = args.demo or problem.GUIDE
    if demo not in problem.HEURISTICS:
        parser.error(f"--demo must be one of {list(problem.HEURISTICS)}")
    if args.quick:
        args.seeds, args.steps = 1, 200_000
    out = HERE / "results" / ("quick" if args.quick else "") / args.problem / demo
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = problem.Config()
    baselines = {name: evaluate(env_cfg, h) for name, h in problem.HEURISTICS.items()}
    for name, b in baselines.items():
        print(f"{name:>25s}: {problem.PLOT['scale'] * b['metric_mean']:.3f}")

    runs = []
    if (args.reuse or args.no_train) and (out / "results.json").exists():   # runs are deterministic
        old = json.loads((out / "results.json").read_text())
        runs = [r for r in old["runs"] if args.no_train or r["ppo_config"]["total_steps"] == args.steps]
        print(f"Reusing {len(runs)} runs from {out / 'results.json'}")
    done = {(r["variant"], r["seed"]) for r in runs}
    jobs = [] if args.no_train else [(args.problem, v, s, args.steps, demo) for v in args.variants
                                     for s in range(args.seeds) if (v, s) not in done]
    jobs.sort(key=lambda j: VARIANTS[j[1]][0] not in SLOW)   # longest first, so the pool does not end on a tail
    if jobs:
        print(f"Training {len(jobs)} runs on {args.workers} workers ...")
        with get_context("spawn").Pool(args.workers) as pool:
            for r in pool.imap_unordered(run_one, jobs):
                print(f"  {r['variant']:>15s} seed {r['seed']}: final {r['final']['metric_mean']:.4f}", flush=True)
                runs.append(r)
    runs.sort(key=lambda r: (list(VARIANTS).index(r["variant"]), r["seed"]))

    rows = summarise(runs, problem, demo)
    (out / "results.json").write_text(json.dumps(
        {"problem": args.problem, "env": asdict(env_cfg), "demo": demo, "baselines": baselines,
         "summary": rows, "runs": runs}, indent=1))
    write_summary(rows, baselines, out / "summary.md", problem, demo)
    # only the full Best Fit run of the toy bin packing feeds the poster, and only if its figures folder exists
    poster = not args.quick and args.problem == "binpack_toy" and demo == "best_fit" and POSTER_FIGURES.is_dir()
    pdf_dir = POSTER_FIGURES if poster else out
    plot_learning_curves(out / "results.json", out, pdf_dir, variants=CLONING)
    plot_learning_curves(out / "results.json", out, pdf_dir, variants=GUIDED, name="learning_curves_guided")
    plot_learning_curves(out / "results.json", out, pdf_dir, variants=ONLINE, name="learning_curves_online")
    plot_learning_curves(out / "results.json", out, pdf_dir, variants=CRAFTER, name="learning_curves_crafter")
    plot_learning_curves(out / "results.json", out, pdf_dir, variants=FADE_FIGURE, name="learning_curves_fade")
    plot_learning_curves(out / "results.json", out, pdf_dir, variants=RELEASE, name="learning_curves_release")
    if args.problem == "binpack_toy":
        from plot_packing import main as plot_packing_example
        plot_packing_example(out.parent, pdf_dir if poster else out.parent)   # same example for every guide
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
