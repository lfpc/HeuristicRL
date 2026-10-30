"""Write report/tables.tex from results/<problem>/<guide>/results.json (run from the repo root or report/)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [  # (folder, column header, scale, unit)
    ("binpack_toy/best_fit", r"\shortstack{Bin packing\\(Best Fit)}", 100, r"\%"),
    ("binpack_toy/sum_of_squares", r"\shortstack{Bin packing\\(Sum of Sq.)}", 100, r"\%"),
    ("binpack_toy/eoh_evolved", r"\shortstack{Bin packing\\(EoH-evolved)}", 100, r"\%"),
    ("knapsack/static_threshold", r"\shortstack{Knapsack\\(static thr.)}", 100, r"\%"),
    ("load_balancing/shortest_expected_delay", r"\shortstack{Load bal.\\(SED)}", 1, ""),
    ("inventory/base_stock", r"\shortstack{Inventory\\(base stock)}", 1, ""),
]
METHODS = [
    ("scratch", "PPO from scratch"),
    ("bc_ppo", r"BC $\to$ PPO (naive)"),
    ("bc_ppo_kl", r"BC $\to$ PPO + KL (50\%)"),
    ("bc_ppo_kl_fast", r"BC $\to$ PPO + KL (20\%)"),
    ("dagger_ppo", r"DAgger $\to$ PPO (naive)"),
    ("dagger_ppo_kl", r"DAgger $\to$ PPO + KL (50\%)"),
    ("hurl", "HuRL (from scratch)"),
    ("bc_hurl", r"BC $\to$ HuRL"),
    ("hepo", "HEPO (from scratch)"),
    ("iql_ppo_kl", r"IQL $\to$ PPO + KL"),
    ("hubl_ppo_kl", r"IQL + HUBL $\to$ PPO + KL"),
    ("kickstart", "Kickstarting (from scratch)"),
    ("bc_anchor", r"BC $\to$ PPO + heur.\ anchor"),
    ("bc_anchor_big", r"BC $\to$ PPO + heur.\ anchor, 4$\times$ batch"),
    ("hubl_anchor", r"IQL + HUBL $\to$ PPO + heur.\ anchor"),
    ("residual", "Residual policy on heuristic"),
    ("jsrl", "Jump-Start RL"),
    ("rollout_dagger_kl", r"Rollout-DAgger $\to$ PPO + KL"),
    ("rollout_dagger_held", r"Rollout-DAgger $\to$ PPO + held KL"),
    ("iql_online", r"IQL offline $\to$ online"),
    ("dqfd", "DQfD"),
    ("crafter_imitate", r"Imitation, annealed 25\% (Crafter)"),
    ("crafter_imitate_c01", "Imitation, constant 0.1 (Crafter)"),
    ("crafter_residual_b2", r"Residual prior $\beta = 2$ (Crafter)"),
    ("crafter_residual_b4", r"Residual prior $\beta = 4$ (Crafter)"),
    ("crafter_residual_b8", r"Residual prior $\beta = 8$ (Crafter)"),
    ("residual_fade", r"Residual prior $\beta = 4$, fading"),
    ("dqfd_fade", "DQfD, fading"),
    ("iql_online_fade", r"IQL offline $\to$ online, fading"),
    ("iql_online_norm", r"IQL offline $\to$ online, fading, normalised AWR"),
    ("iql_online_norm_held", r"IQL offline $\to$ online, normalised AWR, held"),
    ("iql_online_norm_g99", r"IQL offline $\to$ online, fading, normalised AWR, $\gamma = 0.99$"),
    ("bc_imitate", r"BC $\to$ PPO + annealed imitation"),
    ("hubl_imitate", r"IQL + HUBL $\to$ PPO + annealed imitation"),
    ("rollout_dagger_kl_z0", r"Rollout-DAgger (no test) $\to$ PPO + KL"),
    ("bc_qfilter", r"BC $\to$ PPO + Q-filtered imitation"),
    ("hubl_qfilter", r"IQL + HUBL $\to$ PPO + Q-filtered imitation"),
    ("gated", "Performance-gated imitation"),
    ("bc_gated", r"BC $\to$ PPO + perf.-gated imitation"),
    ("hubl_gated", r"IQL + HUBL $\to$ PPO + perf.-gated imitation"),
]


def load():
    return [json.loads((ROOT / "results" / f / "results.json").read_text()) for f, *_ in EXPERIMENTS]


def table(data, key, fmt, caption, label, lower_is_best=True, with_std=False, heuristics=True):
    cols = len(EXPERIMENTS)
    out = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "r" * cols + "}", r"\toprule",
           "Method & " + " & ".join(h for _, h, _, _ in EXPERIMENTS) + r" \\", r"\midrule"]
    if heuristics:
        guide = [f"{s * d['baselines'][d['demo']]['metric_mean']:.2f}" for d, (_, _, s, _) in zip(data, EXPERIMENTS)]
        out.append(r"\textit{Guide heuristic} & " + " & ".join(guide) + r" \\")
        out.append(r"\midrule")
    values = [[d["summary"][m][key] for d in data] for m, _ in METHODS]
    best = []
    for j in range(cols):
        col = [v[j] for v in values if v[j] is not None]
        best.append(min(col) if lower_is_best else max(col))
    for (m, name), row in zip(METHODS, values):
        cells = []
        for j, (d, (_, _, s, _)) in enumerate(zip(data, EXPERIMENTS)):
            v = row[j]
            if v is None:
                cells.append("never")
                continue
            c = fmt(v, s)
            if with_std:
                c += r"\,{\scriptsize$\pm$" + f"{s * d['summary'][m]['final_std']:.2f}" + "}"
            cells.append(r"\textbf{" + c + "}" if v == best[j] else c)
        out.append(f"{name} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(out)


T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
       11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734}


def better(x, y):
    """+1 if mean(x) < mean(y) by a one-sided Welch t-test at 5 %, -1 if mean(x) > mean(y) by the same test, else 0.
    y may be a scalar (one-sample test against a fixed level)."""
    x = np.asarray(x, float)
    vx = x.var(ddof=1) / len(x)
    if np.isscalar(y):
        se, df = np.sqrt(vx), len(x) - 1
        diff = x.mean() - y
    else:
        y = np.asarray(y, float)
        vy = y.var(ddof=1) / len(y)
        se = np.sqrt(vx + vy)
        diff = x.mean() - y.mean()
        df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if se > 0 else 1
    if se == 0:
        return int(np.sign(-diff))
    t = diff / se
    crit = T95[max(1, min(18, int(df)))]          # df rounded down: conservative
    return 1 if t < -crit else (-1 if t > crit else 0)


def finals(d, m):
    return [r["final"]["metric_mean"] for r in d["runs"] if r["variant"] == m]


def scorecard(data):
    """Per method and experiment: Q1 better than PPO from scratch, Q2 beats the guide sooner than PPO from scratch,
    Q3 ends better than the guide. Returns the LaTeX table and the raw marks."""
    sym = {1: r"\ding{51}", 0: r"$\approx$", -1: r"\ding{55}", None: "--"}
    marks = {}
    for m, _ in METHODS:
        row = []
        for d in data:
            q1 = None if m == "scratch" else better(finals(d, m), finals(d, "scratch"))
            s_m, s_0 = d["summary"][m]["steps_to_beat_guide"], d["summary"]["scratch"]["steps_to_beat_guide"]
            if m == "scratch":
                q2 = None
            elif s_m is None:
                q2 = -1
            else:
                q2 = 1 if s_0 is None or s_m < s_0 else (0 if s_m == s_0 else -1)
            q3 = better(finals(d, m), d["baselines"][d["demo"]]["metric_mean"])
            row.append((q1, q2, q3))
        marks[m] = row
    cols = len(EXPERIMENTS)
    out = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
           r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "c" * cols + "ccc}", r"\toprule",
           "Method & " + " & ".join(h for _, h, _, _ in EXPERIMENTS) + r" & Q1 & Q2 & Q3 \\",
           r"\midrule"]
    for m, name in METHODS:
        cells = [" ".join(sym[q] for q in qs) for qs in marks[m]]
        tot = ["--" if m == "scratch" and i < 2 else f"{sum(qs[i] == 1 for qs in marks[m])}/{cols}" for i in range(3)]
        out.append(f"{name} & " + " & ".join(cells) + " & " + " & ".join(tot) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}",
            r"\caption{Scorecard for the three questions. Each cell gives Q1 Q2 Q3 for one experiment: "
            r"Q1 final metric better than PPO from scratch, Q2 beats the guide heuristic in fewer environment steps "
            r"than PPO from scratch, Q3 final metric better than the guide heuristic. "
            r"\ding{51}~yes, \ding{55}~no (significantly worse, slower or never), $\approx$~no significant "
            r"difference. Q1 and Q3 use a one-sided Welch $t$-test at 5\% over the 10 seeds (Q3 against the "
            r"heuristic's value); Q2 compares the median steps of Table~\ref{tab:beat}. Last columns: number of "
            r"experiments with \ding{51}.}",
            r"\label{tab:scorecard}", r"\end{table}", ""]
    return "\n".join(out), marks


def rollout_table(data):
    """Guide vs. its rollout improvement as policies (results/rollout_reference.json), and the simulator cost of the
    Rollout-DAgger labels per run."""
    ref = {(r["problem"], r["guide"]): r for r in json.loads((ROOT / "results" / "rollout_reference.json").read_text())}
    out = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "r" * len(EXPERIMENTS) + "}", r"\toprule",
           " & " + " & ".join(h for _, h, _, _ in EXPERIMENTS) + r" \\", r"\midrule"]
    rows = {"Guide heuristic": [], "Rollout-improved guide": [], r"Best final RL policy (Table~\ref{tab:final})": [],
            "Simulated steps per Rollout-DAgger run": []}
    for d, (_, _, sc, _) in zip(data, EXPERIMENTS):
        r = ref[(d["problem"], d["demo"])]
        rows["Guide heuristic"].append(f"{sc * r['heuristic']:.2f}")
        rows["Rollout-improved guide"].append(f"{sc * r['rollout']:.2f}")
        rows[r"Best final RL policy (Table~\ref{tab:final})"].append(
            f"{sc * min(v['final_mean'] for v in d['summary'].values()):.2f}")
        sim = [run["sim_steps"] for run in d["runs"] if run["variant"] == "rollout_dagger_kl"]
        rows["Simulated steps per Rollout-DAgger run"].append(f"{np.mean(sim) / 1e6:.0f}M")
    out += [f"{k} & " + " & ".join(v) + r" \\" for k, v in rows.items()]
    out += [r"\bottomrule", r"\end{tabular}}",
            r"\caption{The teacher of Rollout-DAgger: each guide and its rollout improvement (32 sampled futures per "
            r"action, switch only at $z = 2$) as policies on their own, on the 256 evaluation episodes "
            r"(\texttt{rollout\_reference.py}); the best final RL policy of any method for comparison; and the "
            r"simulator steps each Rollout-DAgger run spends on its labels, on top of its 1.5M environment steps.}",
            r"\label{tab:rollout}", r"\end{table}", ""]
    return "\n".join(out)


def main():
    data = load()
    units = r"Bin packing: wasted capacity (\%); knapsack: gap to the offline optimum (\%); load balancing: mean " \
            r"sojourn time; inventory: mean cost per period. Lower is better everywhere."
    tex = [
        "% generated by report/make_tables.py -- do not edit by hand",
        scorecard(data)[0],
        table(data, "final_mean", lambda v, s: f"{s * v:.2f}",
              r"Final performance of the trained policy (greedy, 256 fixed evaluation episodes; mean $\pm$ std over "
              r"10 seeds; best per column in bold). " + units, "tab:final", with_std=True),
        table(data, "steps_to_beat_guide", lambda v, _: f"{v / 1e3:.0f}k",
              r"Environment steps until the policy first beats its guide heuristic on the 64 evaluation episodes "
              r"of the learning curves (median over seeds; includes the heuristic's demonstration steps).",
              "tab:beat", heuristics=False),
        rollout_table(data),
    ]
    (Path(__file__).resolve().parent / "tables.tex").write_text("\n".join(tex))
    print("-> report/tables.tex")
    sym = {1: "Y", 0: "~", -1: "N", None: "-"}
    for m, row in scorecard(data)[1].items():
        print(f"{m:15s}", "  ".join("".join(sym[q] for q in qs) for qs in row))


if __name__ == "__main__":
    main()
