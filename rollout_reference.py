"""Each guide heuristic and its rollout improvement (hgrl/rollout.py) as policies on their own,
on the 256 evaluation episodes: the teacher Rollout-DAgger distils. Writes results/rollout_reference.json.

  python rollout_reference.py            # ~70 min for EoH bin packing on one core; the others run alongside
"""
import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

from hgrl.evaluate import evaluate
from hgrl.rollout import RolloutImprovement
from problems import PROBLEMS
from run_experiments import ROLLOUT_SAMPLES

EXPERIMENTS = [("binpack", "best_fit"), ("binpack", "sum_of_squares"), ("binpack", "eoh_evolved"),
               ("binpack_toy", "best_fit"), ("binpack_toy", "sum_of_squares"), ("binpack_toy", "eoh_evolved"),
               ("knapsack", "static_threshold"), ("load_balancing", "shortest_expected_delay"),
               ("inventory", "base_stock")]
OUT = Path(__file__).resolve().parent / "results" / "rollout_reference.json"


def job(exp):
    problem = PROBLEMS[exp[0]]
    cfg, heuristic = problem.Config(), problem.HEURISTICS[exp[1]]
    t0 = time.time()
    improved = RolloutImprovement(heuristic, n_samples=ROLLOUT_SAMPLES, seed=1)
    return {"problem": exp[0], "guide": exp[1], "heuristic": evaluate(cfg, heuristic)["metric_mean"],
            "rollout": evaluate(cfg, improved)["metric_mean"], "n_samples": ROLLOUT_SAMPLES, "z": improved.z,
            "sim_steps": improved.sim_steps, "seconds": time.time() - t0}


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter).parse_args()
    with get_context("spawn").Pool(len(EXPERIMENTS)) as pool:
        rows = pool.map(job, EXPERIMENTS)
    OUT.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['problem']:15s} {r['guide']:24s} {r['heuristic']:.4f} -> {r['rollout']:.4f}")


if __name__ == "__main__":
    main()
