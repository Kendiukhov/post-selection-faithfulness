"""Build every figure and every auto-generated LaTeX fragment for the paper.

Numbers quoted in the text are never typed by hand: they are emitted here into
``paper/generated/numbers.tex`` as LaTeX macros, so the manuscript and the
results cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import os

# BLAS thread pools spin-wait; on a busy machine that can make a small mat-mul
# a thousand times slower than the single-threaded version.  The heavy linear
# algebra here is either tiny or runs on the accelerator, so pin the pools to
# one thread each.  Must happen before numpy is imported.
for _v in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PALETTE = {
    "naive": "#B3282D",
    "union-hoeffding": "#8C8C8C",
    "union-eb": "#4C72B0",
    "union-best": "#3A5FCD",
    "occam": "#55A868",
    "boot-max": "#1F1F1F",
    "boot-floored": "#00707A",
    "split-eb": "#B07C4F",
    "boot-max-cluster": "#8452A1",
    "split": "#DD8452",
    "conditional": "#937860",
    "hybrid": "#DA8BC3",
}
NICE = {
    "naive": "naive ($\\pm z\\cdot$SE)",
    "union-hoeffding": "union (Hoeffding)",
    "union-eb": "union (emp. Bernstein)",
    "union-best": "union (best of two)",
    "occam": "Occam (size prior)",
    "boot-max": "bootstrap-$t$ band",
    "boot-floored": "bootstrap band + floor",
    "split-eb": "splitting (finite-sample)",
    "boot-max-cluster": "cluster bootstrap band",
    "split": "sample splitting",
    "conditional": "conditional (selective)",
    "hybrid": "hybrid selective",
}
SHORT = {
    "naive": "naive",
    "union-hoeffding": "Hoeffding",
    "union-eb": "emp. Bernstein",
    "union-best": "union bound",
    "occam": "Occam",
    "boot-max": "bootstrap-$t$",
    "boot-floored": "boot + floor",
    "boot-max-cluster": "cluster boot",
    "split": "splitting",
    "split-eb": "splitting (f.s.)",
    "conditional": "conditional",
    "hybrid": "hybrid",
}
ORDER = [
    "naive", "union-hoeffding", "union-eb", "union-best", "occam",
    "boot-max", "boot-floored", "boot-max-cluster", "split", "split-eb",
    "conditional", "hybrid",
]

TEXTWIDTH = 6.5  # inches; figures are drawn at their final printed size

plt.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

MACROS: dict[str, str] = {}


def mac(name: str, value) -> None:
    MACROS[name] = str(value)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"  # LaTeX-escaped: consumed by numbers.tex, not by matplotlib


# ---------------------------------------------------------------------------
# Figure 1: anatomy of the problem (synthetic)
# ---------------------------------------------------------------------------


def fig_anatomy(syn: dict, outdir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.25), constrained_layout=True)

    # (a) exact naive coverage vs m
    ax = axes[0]
    A = syn["A"]
    ms = np.array(A["ms"])
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(A["rhos"])))
    for col, r in zip(cols, A["rhos"]):
        ax.plot(ms, A["exact"][str(r)], color=col, lw=1.5, label=f"$\\rho={r}$")
        ax.plot(A["sim_ms"], A["sim"][str(r)], "o", ms=2.6, mfc="none", mew=0.8, color=col)
    ax.axhline(0.95, color="k", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("hypotheses searched")
    ax.set_ylabel("coverage of the naive 95% bound")
    ax.set_title("(a) The usual error bar fails")
    # every curve starts at 0.95 and falls, so a single legend row above 1.0 is
    # the only place that no curve can reach
    ax.set_ylim(-0.03, 1.20)
    ax.legend(frameon=False, fontsize=5.2, loc="upper center", ncol=4,
              handlelength=0.9, columnspacing=0.45, handletextpad=0.35,
              borderpad=0.1)
    ax.text(0.97, 0.62, "lines: exact\ncircles: simulation", transform=ax.transAxes,
            fontsize=5.8, color="0.35", ha="right", va="top")

    # (b) optimism scaling
    ax = axes[1]
    rows = [r for r in syn["B"]["rows"] if r["rho"] == 0.0]
    ms_b = sorted({r["m"] for r in rows})
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(ms_b)))
    for col, m in zip(cmap, ms_b):
        rr = sorted([r for r in rows if r["m"] == m], key=lambda x: x["n"])
        ax.plot([r["n"] for r in rr], [r["optimism"] for r in rr], "o-", ms=2.4,
                color=col, lw=1.2, label=f"{m}")
        ax.plot([r["n"] for r in rr], [r["predicted_iid"] for r in rr], ":", color=col, lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("interventions $n$")
    ax.set_ylabel("reported $-$ true faithfulness")
    ax.set_title("(b) Optimism follows the theory")
    # open a band below the lowest curve so the legend cannot touch any of them
    lo = min(r["optimism"] for r in rows)
    hi = max(max(r["optimism"], r["predicted_iid"]) for r in rows)
    ax.set_ylim(lo / 3.4, hi * 1.25)
    leg = ax.legend(frameon=False, fontsize=5.6, ncol=2, title="hypotheses",
                    handlelength=1.2, columnspacing=0.8, borderpad=0.1,
                    labelspacing=0.25, handletextpad=0.5, loc="lower left")
    leg.get_title().set_fontsize(5.6)
    ax.text(0.97, 0.95, "dotted: $\\sigma\\sqrt{2\\log m/n}$", transform=ax.transAxes,
            fontsize=5.8, color="0.35", ha="right", va="top")

    # (c) how much each method certifies, as the hypotheses become correlated
    ax = axes[2]
    rows = [r for r in syn["C"]["rows"] if r.get("method") not in (None, "_diag", "_selection")]
    rhos = sorted({r["rho"] for r in rows if r["shape"] == "flat"})
    show = ["boot-max", "boot-floored", "union-best", "split", "split-eb", "hybrid"]
    for nm in show:
        pts = [
            next((r for r in rows if r["shape"] == "flat" and r["m"] == 200
                  and r["n"] == 1000 and r["rho"] == rho and r["method"] == nm), None)
            for rho in rhos
        ]
        if any(p is None for p in pts):
            continue
        ax.plot(rhos, [p["mean_lcb"] for p in pts], "o-", ms=2.8, lw=1.2,
                color=PALETTE.get(nm, "0.5"), label=SHORT.get(nm, nm))
    nv = [
        next((r for r in rows if r["shape"] == "flat" and r["m"] == 200 and r["n"] == 1000
              and r["rho"] == rho and r["method"] == "naive"), None)
        for rho in rhos
    ]
    if all(v is not None for v in nv):
        ax.plot(rhos, [v["mean_lcb"] for v in nv], "x--", ms=4, lw=1.1,
                color=PALETTE["naive"], label="naive (invalid)")
    ax.axhline(0.80, color="k", ls=":", lw=1.0)
    ax.text(0.02, 0.801, "true faithfulness", fontsize=5.8, color="0.25", va="bottom")
    ax.set_xlabel("correlation between hypotheses $\\rho$")
    ax.set_ylabel("mean certified lower bound")
    ax.set_title("(c) What each method certifies")
    ax.set_xticks(rhos)
    # the curves occupy a narrow band; extend the axis downwards so the legend
    # sits in empty space instead of on top of them
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo - 0.42 * (yhi - ylo), yhi)
    ax.legend(frameon=False, fontsize=5.4, ncol=2, handlelength=1.4,
              columnspacing=0.7, borderpad=0.1, labelspacing=0.25,
              handletextpad=0.5, loc="lower left")

    fig.savefig(os.path.join(outdir, "fig_anatomy.pdf"))
    plt.close(fig)

    exact0 = dict(zip(A["ms"], A["exact"]["0.0"]))
    exact9 = dict(zip(A["ms"], A["exact"]["0.9"]))
    m100 = min(A["ms"], key=lambda x: abs(x - 100))
    mac("MyNaiveCovHundred", pct(exact0[m100]))
    mac("MyNaiveCovHundredRhoNine", pct(exact9[m100]))


# ---------------------------------------------------------------------------
# Figure 2: tiny transformers
# ---------------------------------------------------------------------------


def fig_tiny(tiny: dict, outdir: str) -> None:
    tags = [t for t in ["tt_a", "tt_a_local", "tt_c", "tt_b"] if t in tiny]
    titles = {
        "tt_a": "(a) a clear winner",
        "tt_a_local": "(b) near-ties",
        "tt_c": "(c) nothing is faithful",
        "tt_b": "(d) head subsets",
    }
    show = ["naive", "union-best", "boot-max", "boot-floored", "split", "split-eb", "hybrid"]
    fig, axes = plt.subplots(2, len(tags), figsize=(TEXTWIDTH, 3.4),
                             sharex=True, constrained_layout=True)
    for k, tag in enumerate(tags):
        rows = tiny[tag]["rows"]
        ax = axes[0, k]
        for nm in show:
            rr = sorted([r for r in rows if r["method"] == nm], key=lambda x: x["n"])
            if not rr:
                continue
            ax.plot([r["n"] for r in rr], [r["coverage"] for r in rr], "o-", ms=2.2, lw=1.0,
                    color=PALETTE[nm], label=SHORT.get(nm, nm))
        ax.axhline(0.95, color="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.06)
        ax.set_title(titles.get(tag, tag))
        if k == 0:
            ax.set_ylabel("coverage")

        ax = axes[1, k]
        sel = sorted([r for r in rows if r["method"] == "_selection"], key=lambda x: x["n"])
        vals = []
        for nm in show:
            rr = sorted([r for r in rows if r["method"] == nm], key=lambda x: x["n"])
            if not rr:
                continue
            ax.plot([r["n"] for r in rr], [r["mean_lcb"] for r in rr], "o-", ms=2.2, lw=1.0,
                    color=PALETTE[nm])
            vals += [r["mean_lcb"] for r in rr]
        if sel:
            ax.plot([r["n"] for r in sel], [r["mean_theta_selected"] for r in sel], "k:",
                    lw=1.3, label="truth")
            ax.plot([r["n"] for r in sel], [r["mean_point"] for r in sel], color="0.45",
                    ls=(0, (4, 2)), lw=1.1, label="naive point estimate")
            vals += [r["mean_theta_selected"] for r in sel] + [r["mean_point"] for r in sel]
        hi = max(vals) + 0.02
        lo = max(min(vals), min(vals) if min(vals) > -0.2 else -0.05) - 0.02
        ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        if k == 0:
            ax.set_ylabel("certified lower bound")
        ax.set_xlabel("interventions $n$")
        if k == len(tags) - 1:
            ax.legend(frameon=False, fontsize=5.4, loc="lower right", borderpad=0.1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.2, ncol=7,
               loc="outside lower center", handlelength=1.3, columnspacing=1.0)
    fig.savefig(os.path.join(outdir, "fig_tiny.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: GPT-2 IOI
# ---------------------------------------------------------------------------


def fig_ioi(ioi: dict, outdir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.45), constrained_layout=True)

    # (a) coverage vs n -- methods shared with panel (c) go in one figure legend
    ax = axes[0]
    rows = [r for r in ioi["iid"]["rows"] if r.get("selection") == "argmax"]
    shown = ["naive", "split", "hybrid", "union-best", "boot-max", "boot-floored"]
    ns_a = sorted({r["n"] for r in rows})
    for nm in shown:
        rr = sorted([r for r in rows if r["method"] == nm], key=lambda x: x["n"])
        if not rr:
            continue
        ax.plot([r["n"] for r in rr], [r["coverage"] for r in rr], "o-", ms=2.6, lw=1.2,
                color=PALETTE[nm], label=SHORT.get(nm, nm))
    ax.axhline(0.95, color="k", ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(ns_a)
    ax.set_xticklabels([str(n) for n in ns_a])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.09)
    ax.set_xlabel("interventions $n$")
    ax.set_ylabel("coverage")
    ax.set_title("(a) Coverage on IOI")

    # (b) certified Pareto frontier
    ax = axes[1]
    fr = [r for r in ioi["frontier"]["rows"] if r["k"] <= 30]
    ks = [r["k"] for r in fr]
    curves = [
        ("naive point estimate", [r["naive_point"] for r in fr], PALETTE["naive"], "-", 1.4),
        ("truth", [r["truth"] for r in fr], "k", ":", 1.4),
        ("simultaneous band", [r["boot_lcb"] for r in fr], PALETTE["boot-max"], "-", 1.4),
        ("Occam bound", [r["occam_lcb"] for r in fr], PALETTE["occam"], "-", 1.1),
    ]
    for lab, y, col, ls, lw in curves:
        ax.plot(ks, y, color=col, ls=ls, lw=lw, label=lab)
    ax.set_xlabel("circuit size budget (heads)")
    ax.set_ylabel("faithfulness")
    ax.set_title("(b) The whole frontier at once")
    ax.set_xlim(-1, 31)
    lo = min(min(y) for _, y, _, _, _ in curves)
    ax.set_ylim(lo - 0.02, 1.03)
    # the curves all rise to the top right, so the lower right corner is free
    ax.legend(frameon=False, fontsize=5.6, loc="lower right", ncol=1,
              handlelength=1.4, borderpad=0.1, labelspacing=0.3)

    # (c) clustered sampling
    ax = axes[2]
    cl = ioi["cluster"]["rows"]
    ns = sorted({r["n"] for r in cl})
    width = 0.36
    for j, nm in enumerate(["boot-max", "boot-max-cluster"]):
        vals = [next((r["coverage"] for r in cl if r["n"] == n and r["method"] == nm), np.nan)
                for n in ns]
        ax.bar(np.arange(len(ns)) + (j - 0.5) * width, vals, width,
               color=PALETTE[nm], label=SHORT.get(nm, nm))
    nv = [next((r["coverage"] for r in cl if r["n"] == n and r["method"] == "naive"), np.nan)
          for n in ns]
    ax.plot(np.arange(len(ns)), nv, "o--", ms=3.4, lw=1.1, color=PALETTE["naive"],
            label=SHORT["naive"])
    ax.axhline(0.95, color="k", ls="--", lw=0.9)
    ax.set_xticks(np.arange(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("prompts $n$, drawn by template")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.09)
    ax.set_title("(c) Templates are the unit")

    # one legend for the whole figure, below the panels: panels (a) and (c) draw
    # from the same set of methods, so a per-panel legend only invites overlap
    handles, labels = [], []
    for a in (axes[0], axes[2]):
        for h, lb in zip(*a.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    fig.legend(handles, labels, frameon=False, fontsize=6.2, ncol=7,
               loc="outside lower center", handlelength=1.4, columnspacing=1.1)
    fig.savefig(os.path.join(outdir, "fig_ioi.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: adaptive greedy search
# ---------------------------------------------------------------------------


def fig_greedy(g: dict, outdir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.35), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    sizes = np.array(g["sizes"])
    sc = np.array(g["search_curve"])
    ec = np.array(g["eval_curve"])

    # (a) the two curves, labelled directly: a legend box would sit on top of them
    ax = axes[0]
    ax.fill_between(sizes, ec, sc, where=sc >= ec, color=PALETTE["naive"], alpha=0.25, lw=0)
    ax.plot(sizes, sc, color=PALETTE["naive"], lw=1.4)
    ax.plot(sizes, ec, "k:", lw=1.4)
    ps = g["p_star"]
    ax.plot([sizes[ps]], [sc[ps]], "o", ms=4, color="k", mfc="white", mew=1.2, zorder=5)
    ax.invert_xaxis()
    ax.set_ylim(min(ec.min(), sc.min()) - 0.05, max(sc) + 0.28)
    ax.annotate("scored on the search set", (sizes[np.argmax(sc)], max(sc)),
                textcoords="offset points", xytext=(0, 14), ha="center",
                fontsize=6.0, color=PALETTE["naive"])
    # anchor in the empty region under the plateau, where the curves are far
    # away and nothing else is drawn
    j = int(np.argmin(np.abs(sizes - 46)))
    ax.annotate("scored on held-out\ninterventions", (sizes[j], ec[j]),
                textcoords="offset points", xytext=(0, -48), ha="center", va="top",
                fontsize=6.0, color="0.15",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4",
                                shrinkA=1, shrinkB=3))
    # to the left of the marker: everything above it is on the falling curve
    ax.annotate(f"returned circuit\n({g['final_size']} heads)", (sizes[ps], sc[ps]),
                textcoords="offset points", xytext=(-46, -1), ha="right", va="center",
                fontsize=6.0, color="0.25",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4",
                                shrinkA=1, shrinkB=4))
    ax.set_xlabel("heads remaining")
    ax.set_ylabel("fraction of logit difference\nrecovered")
    ax.set_title("(a) A pruner overfits its own data")

    # (b) horizontal bars; values sit inside the bars, clear of the truth line
    ax = axes[1]
    short = {
        "naive bound on the search data": "naive, on the search data",
        "held-out, 200 fresh interventions": "held out, 200 interventions",
        "held-out, 1200 fresh interventions": "held out, 1200 interventions",
    }
    labels, vals, cols = [], [], []
    for r in g["bounds"]:
        nm = r["name"]
        if "could have reached" in nm:
            nm = "uniform over the reachable set"
        labels.append(short.get(nm, nm))
        vals.append(r["lcb"])
        cols.append(r.get("color", "#4C72B0"))
    ypos = np.arange(len(vals))
    ax.barh(ypos, vals, color=cols, height=0.6)
    for y, v in zip(ypos, vals):
        ax.text(v - 0.012, y, f"{v:.3f}", va="center", ha="right", fontsize=6.0,
                color="white", fontweight="bold")
    ax.axvline(g["truth"], color="k", ls=":", lw=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.invert_yaxis()
    # a strip above the first bar, so the label for the truth line sits clear of it
    ax.set_ylim(len(vals) - 0.45, -1.00)
    ax.annotate("truth", (g["truth"], -0.66), textcoords="offset points",
                xytext=(-3, 0), fontsize=6.0, color="0.25", va="center", ha="right")
    ax.set_xlim(0, max(max(vals), g["truth"]) * 1.10)
    ax.set_xlabel("certified faithfulness of the returned circuit")
    ax.set_title("(b) What survives an adaptive search")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(outdir, "fig_greedy.pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--gendir", default="paper/generated")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.gendir, exist_ok=True)

    def load(p):
        p = os.path.join(args.results, p)
        return json.load(open(p)) if os.path.exists(p) else None

    syn = load("synthetic/synthetic.json")
    tiny = load("tiny/tiny_analysis.json")
    ioi = load("ioi/ioi_analysis.json")
    greedy = load("ioi/greedy_analysis.json")

    if syn:
        fig_anatomy(syn, args.figdir)
        print("wrote fig_anatomy.pdf")
    if tiny:
        fig_tiny(tiny, args.figdir)
        print("wrote fig_tiny.pdf")
    if ioi:
        fig_ioi(ioi, args.figdir)
        print("wrote fig_ioi.pdf")
    if greedy:
        fig_greedy(greedy, args.figdir)
        print("wrote fig_greedy.pdf")

    adaptive = load("adaptive/adaptive.json")
    from make_tables import build_all  # noqa: E402

    build_all(syn, tiny, ioi, greedy, adaptive, args.gendir, MACROS)

    # Any \MyXxx macro used in the manuscript but not produced by a finished
    # experiment gets a loud placeholder, so a partial build still compiles and
    # the missing numbers are impossible to overlook.
    import glob as _glob
    import re as _re

    used = set()
    for tex in _glob.glob(os.path.join(os.path.dirname(args.gendir), "*.tex")) + _glob.glob(
        os.path.join(args.gendir, "*.tex")
    ):
        if os.path.basename(tex) == "numbers.tex":
            continue  # would re-detect its own placeholders
        used |= set(_re.findall(r"\\(My[A-Za-z]+)", open(tex).read()))
    # stubs for \input{generated/...} targets whose experiment has not been run,
    # and for figures that have not been produced, so the manuscript still builds
    for tex in _glob.glob(os.path.join(os.path.dirname(args.gendir), "*.tex")):
        body = open(tex).read()
        for name in _re.findall(r"\\input\{generated/([A-Za-z_0-9]+)\}", body):
            path = os.path.join(args.gendir, name + ".tex")
            if not os.path.exists(path):
                open(path, "w").write(
                    "\\begin{center}\\textbf{?? " + name.replace("_", "\\_")
                    + " not generated ??}\\end{center}\n"
                )
        for name in _re.findall(r"\\includegraphics\[[^\]]*\]\{\.\./figures/([A-Za-z_0-9]+)\.pdf\}", body):
            path = os.path.join(args.figdir, name + ".pdf")
            if not os.path.exists(path):
                fg = plt.figure(figsize=(6.5, 1.0))
                fg.text(0.5, 0.5, f"?? {name} not generated ??", ha="center", va="center")
                fg.savefig(path)
                plt.close(fg)

    missing = sorted(used - set(MACROS))
    with open(os.path.join(args.gendir, "numbers.tex"), "w") as f:
        f.write("% auto-generated by experiments/make_figures.py -- do not edit\n")
        for k, v in sorted(MACROS.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        if missing:
            f.write("% placeholders for experiments that have not been run\n")
            for k in missing:
                f.write(f"\\newcommand{{\\{k}}}{{\\textbf{{??}}}}\n")
    print(f"wrote {len(MACROS)} macros to numbers.tex"
          + (f"; {len(missing)} placeholders: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
