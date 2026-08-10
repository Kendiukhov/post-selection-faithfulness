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
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.15))

    # (a) exact naive coverage vs m
    ax = axes[0]
    A = syn["A"]
    ms = np.array(A["ms"])
    for r in A["rhos"]:
        ax.plot(ms, A["exact"][str(r)], label=f"$\\rho={r}$", lw=1.6)
    for r in A["rhos"]:
        ax.plot(A["sim_ms"], A["sim"][str(r)], "o", ms=3.2, mfc="none", color="k", alpha=0.55)
    ax.axhline(0.95, color="k", ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("number of hypotheses searched")
    ax.set_ylabel("coverage of the naive 95\\% bound")
    ax.set_title("(a) The usual error bar stops working")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    ax.text(0.97, 0.93, "circles: simulation\nlines: exact", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="0.3")

    # (b) optimism scaling
    ax = axes[1]
    rows = [r for r in syn["B"]["rows"] if r["rho"] == 0.0]
    ms_b = sorted({r["m"] for r in rows})
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(ms_b)))
    for col, m in zip(cmap, ms_b):
        rr = sorted([r for r in rows if r["m"] == m], key=lambda x: x["n"])
        ax.plot([r["n"] for r in rr], [r["optimism"] for r in rr], "o-", ms=3,
                color=col, lw=1.3, label=f"$m={m}$")
        ax.plot([r["n"] for r in rr], [r["predicted_iid"] for r in rr], ":", color=col, lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of interventions $n$")
    ax.set_ylabel("optimism (reported $-$ true faithfulness)")
    ax.set_title("(b) Optimism follows the predicted law")
    ax.legend(frameon=False, fontsize=7, ncol=1)
    ax.text(0.03, 0.06, "dotted: theory", transform=ax.transAxes, fontsize=7, color="0.3")

    # (c) certified value vs coverage across regimes
    ax = axes[2]
    rows = [r for r in syn["C"]["rows"] if r.get("method") not in (None, "_diag", "_selection")]
    sel = [r for r in rows if r["n"] == 1000 and r["shape"] == "flat" and r["m"] == 200]
    names = [r["method"] for r in sel]
    xs = [r["coverage"] for r in sel]
    ys = [r["mean_lcb"] for r in sel]
    for nm, x, y in zip(names, xs, ys):
        ax.scatter(x, y, s=34, color=PALETTE.get(nm, "k"), zorder=3,
                   marker="X" if nm == "naive" else "o")
        ax.annotate(NICE.get(nm, nm), (x, y), textcoords="offset points",
                    xytext=(5, 3), fontsize=6.6, color=PALETTE.get(nm, "k"))
    ax.axvline(0.95, color="k", ls="--", lw=0.9)
    ax.set_xlabel("coverage (nominal 0.95)")
    ax.set_ylabel("mean certified lower bound")
    ax.set_title("(c) Valid methods, ranked by what they certify")
    ax.set_xlim(-0.03, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_anatomy.pdf"))
    plt.close(fig)

    # macros
    exact0 = dict(zip(A["ms"], A["exact"]["0.0"]))
    exact9 = dict(zip(A["ms"], A["exact"]["0.9"]))
    m100 = min(A["ms"], key=lambda x: abs(x - 100))
    mac("MyNaiveCovHundred", pct(exact0[m100]))
    mac("MyNaiveCovHundredRhoNine", pct(exact9[m100]))


# ---------------------------------------------------------------------------
# Figure 2: tiny transformers
# ---------------------------------------------------------------------------


def fig_tiny(tiny: dict, outdir: str) -> None:
    tags = [t for t in ["tt_a", "tt_a_local", "tt_a_iit", "tt_c", "tt_b"] if t in tiny]
    titles = {
        "tt_a": "(a) TT-A equality: a clear winner",
        "tt_a_local": "(b) TT-A equality: near-ties",
        "tt_a_iit": "(c) TT-A-IIT: planted alignment",
        "tt_c": "(d) TT-C arithmetic: nothing faithful",
        "tt_b": "(e) TT-B induction: head subsets",
    }
    fig, axes = plt.subplots(2, len(tags), figsize=(TEXTWIDTH, 3.6), sharex=True)
    if len(tags) == 1:
        axes = axes[:, None]
    for k, tag in enumerate(tags):
        rows = tiny[tag]["rows"]
        ns = sorted({r["n"] for r in rows if r["method"] != "_selection"})
        # top: coverage
        ax = axes[0, k]
        for nm in ORDER:
            rr = sorted([r for r in rows if r["method"] == nm], key=lambda x: x["n"])
            if not rr:
                continue
            ax.plot([r["n"] for r in rr], [r["coverage"] for r in rr], "o-", ms=2.8, lw=1.2,
                    color=PALETTE[nm], label=NICE[nm])
        ax.axhline(0.95, color="k", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(titles.get(tag, tag))
        if k == 0:
            ax.set_ylabel("coverage")
        # bottom: certified value
        ax = axes[1, k]
        for nm in ORDER:
            rr = sorted([r for r in rows if r["method"] == nm], key=lambda x: x["n"])
            if not rr:
                continue
            ax.plot([r["n"] for r in rr], [r["mean_lcb"] for r in rr], "o-", ms=2.8, lw=1.2,
                    color=PALETTE[nm])
        sel = sorted([r for r in rows if r["method"] == "_selection"], key=lambda x: x["n"])
        if sel:
            ax.plot([r["n"] for r in sel], [r["mean_theta_selected"] for r in sel], "k:",
                    lw=1.4, label="truth $\\theta(\\hat c)$")
            ax.plot([r["n"] for r in sel], [r["mean_point"] for r in sel], color="0.45",
                    ls=(0, (4, 2)), lw=1.2, label="naive point estimate")
        ax.set_xscale("log")
        ax.set_xlabel("interventions $n$")
        if k == 0:
            ax.set_ylabel("certified lower bound")
        if k == len(tags) - 1:
            ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.2, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_tiny.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: GPT-2 IOI
# ---------------------------------------------------------------------------


def fig_ioi(ioi: dict, outdir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.25))

    # (a) coverage and certified value vs n
    ax = axes[0]
    rows = [r for r in ioi["iid"]["rows"] if r.get("selection") == "argmax"]
    for nm in ORDER:
        rr = sorted([r for r in rows if r["method"] == nm], key=lambda x: x["n"])
        if not rr:
            continue
        ax.plot([r["n"] for r in rr], [r["coverage"] for r in rr], "o-", ms=3, lw=1.3,
                color=PALETTE[nm], label=NICE[nm])
    ax.axhline(0.95, color="k", ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("interventions $n$")
    ax.set_ylabel("coverage")
    ax.set_title("(a) Coverage on IOI, GPT-2 small")
    ax.legend(frameon=False, fontsize=6.4, ncol=2, loc="lower right")

    # (b) certified Pareto frontier
    ax = axes[1]
    fr = ioi["frontier"]["rows"]
    ks = [r["k"] for r in fr]
    ax.plot(ks, [r["naive_point"] for r in fr], color="#B3282D", lw=1.5,
            label="naive point estimate")
    ax.plot(ks, [r["truth"] for r in fr], "k:", lw=1.5, label="truth")
    ax.plot(ks, [r["boot_lcb"] for r in fr], color="#1F1F1F", lw=1.5,
            label="simultaneous band")
    ax.plot(ks, [r["occam_lcb"] for r in fr], color="#55A868", lw=1.2, label="Occam bound")
    ax.set_xlabel("circuit size budget (number of heads)")
    ax.set_ylabel("faithfulness (IO ranked above S)")
    ax.set_title("(b) The whole frontier, certified at once")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    # (c) clustered vs iid
    ax = axes[2]
    cl = ioi["cluster"]["rows"]
    ns = sorted({r["n"] for r in cl})
    width = 0.36
    meths = ["boot-max", "boot-max-cluster"]
    for j, nm in enumerate(meths):
        vals = [next(r["coverage"] for r in cl if r["n"] == n and r["method"] == nm) for n in ns]
        ax.bar(np.arange(len(ns)) + (j - 0.5) * width, vals, width,
               color=PALETTE[nm], label=NICE[nm])
    ax.axhline(0.95, color="k", ls="--", lw=0.9)
    ax.set_xticks(np.arange(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("prompts $n$ (drawn template-by-template)")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) Templates, not prompts, are the unit")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_ioi.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: adaptive greedy search
# ---------------------------------------------------------------------------


def fig_greedy(g: dict, outdir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4))
    ax = axes[0]
    sizes = g["sizes"]
    ax.plot(sizes, g["search_curve"], color="#B3282D", lw=1.5,
            label="score on the search set")
    ax.plot(sizes, g["eval_curve"], "k:", lw=1.5, label="score on held-out interventions")
    ax.invert_xaxis()
    ax.set_xlabel("heads remaining")
    ax.set_ylabel("fraction of logit difference recovered")
    ax.set_title("(a) A greedy search overfits its own data")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    labels = [r["name"] for r in g["bounds"]]
    vals = [r["lcb"] for r in g["bounds"]]
    cols = [r.get("color", "#4C72B0") for r in g["bounds"]]
    ax.barh(range(len(vals)), vals, color=cols)
    ax.axvline(g["truth"], color="k", ls=":", lw=1.4)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("certified faithfulness of the returned circuit")
    ax.set_title("(b) What survives an adaptive search")
    fig.tight_layout()
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

    with open(os.path.join(args.gendir, "numbers.tex"), "w") as f:
        f.write("% auto-generated by experiments/make_figures.py -- do not edit\n")
        for k, v in sorted(MACROS.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(MACROS)} macros to numbers.tex")


if __name__ == "__main__":
    main()
