"""Write report/tables.tex from results/<problem>/<guide>/results.json (run from the repo root or report/)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [  # (folder, column header, scale, unit)
    ("binpack/best_fit", r"\shortstack{Bin packing\\(Best Fit)}", 100, r"\%"),
    ("binpack/sum_of_squares", r"\shortstack{Bin packing\\(Sum of Sq.)}", 100, r"\%"),
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
]


def load():
    return [json.loads((ROOT / "results" / f / "results.json").read_text()) for f, *_ in EXPERIMENTS]


def table(data, key, fmt, caption, label, lower_is_best=True, with_std=False, heuristics=True):
    cols = len(EXPERIMENTS)
    out = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{l" + "r" * cols + "}", r"\toprule",
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
    out += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(out)


T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}


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
    crit = T95[max(1, min(10, int(df)))]          # df rounded down: conservative
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
           r"\begin{tabular}{l" + "c" * cols + "ccc}", r"\toprule",
           "Method & " + " & ".join(h for _, h, _, _ in EXPERIMENTS) + r" & Q1 & Q2 & Q3 \\",
           r"\midrule"]
    for m, name in METHODS:
        cells = [" ".join(sym[q] for q in qs) for qs in marks[m]]
        tot = ["--" if m == "scratch" and i < 2 else f"{sum(qs[i] == 1 for qs in marks[m])}/{cols}" for i in range(3)]
        out.append(f"{name} & " + " & ".join(cells) + " & " + " & ".join(tot) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Scorecard for the three questions. Each cell gives Q1 Q2 Q3 for one experiment: "
            r"Q1 final metric better than PPO from scratch, Q2 beats the guide heuristic in fewer environment steps "
            r"than PPO from scratch, Q3 final metric better than the guide heuristic. "
            r"\ding{51}~yes, \ding{55}~no (significantly worse, slower or never), $\approx$~no significant "
            r"difference. Q1 and Q3 use a one-sided Welch $t$-test at 5\% over the 5 seeds (Q3 against the "
            r"heuristic's value); Q2 compares the median steps of Table~\ref{tab:beat}. Last columns: number of "
            r"experiments with \ding{51}.}",
            r"\label{tab:scorecard}", r"\end{table}", ""]
    return "\n".join(out), marks


def main():
    data = load()
    units = r"Bin packing: wasted capacity (\%); knapsack: gap to the offline optimum (\%); load balancing: mean " \
            r"sojourn time; inventory: mean cost per period. Lower is better everywhere."
    tex = [
        "% generated by report/make_tables.py -- do not edit by hand",
        scorecard(data)[0],
        table(data, "final_mean", lambda v, s: f"{s * v:.2f}",
              r"Final performance of the trained policy (greedy, 256 fixed evaluation episodes; mean $\pm$ std over "
              r"5 seeds; best per column in bold). " + units, "tab:final", with_std=True),
        table(data, "steps_to_beat_guide", lambda v, _: f"{v / 1e3:.0f}k",
              r"Environment steps until the policy first beats its guide heuristic on the 64 evaluation episodes "
              r"of the learning curves (median over seeds; includes the heuristic's demonstration steps).",
              "tab:beat", heuristics=False),
    ]
    (Path(__file__).resolve().parent / "tables.tex").write_text("\n".join(tex))
    print("-> report/tables.tex")
    sym = {1: "Y", 0: "~", -1: "N", None: "-"}
    for m, row in scorecard(data)[1].items():
        print(f"{m:15s}", "  ".join("".join(sym[q] for q in qs) for qs in row))


if __name__ == "__main__":
    main()
