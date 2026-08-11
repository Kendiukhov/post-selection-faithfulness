"""LaTeX tables and macros generated directly from the results JSON files."""

from __future__ import annotations

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

import numpy as np

NICE = {
    "naive": "naive $\\pm z\\,\\mathrm{SE}$ \\emph{(invalid)}",
    "boot-floored": "bootstrap band, finite-sample floor",
    "split-eb": "sample splitting, finite-sample",
    "union-hoeffding": "union bound, Hoeffding",
    "union-eb": "union bound, empirical Bernstein",
    "union-best": "union bound, best of two",
    "occam": "Occam bound (size prior)",
    "boot-max": "bootstrap simultaneous band",
    "boot-max-cluster": "cluster bootstrap band",
    "split": "sample splitting ($50/50$)",
    "conditional": "conditional (selective)",
    "hybrid": "hybrid selective",
}
ORDER = [
    "naive", "union-hoeffding", "union-eb", "union-best", "occam",
    "boot-max", "boot-floored", "split", "split-eb", "conditional", "hybrid",
]


def _fmt(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{d}f}"


def _cov(row):
    c, se = row["coverage"], row["coverage_se"]
    star = "" if c >= 0.95 - 2 * se else "$^{\\dagger}$"
    return f"{c:.3f}{star}"


def table_main(ioi, gendir, mac):
    """Headline table: GPT-2 IOI at n = 1000, i.i.d. instance sampling."""
    if ioi is None:
        return
    n = 1000
    rows = [r for r in ioi["iid"]["rows"]
            if r.get("n") == n and r.get("selection") == "argmax"]
    sel = next((r for r in rows if r["method"] == "_selection"), None)
    body = []
    for nm in ORDER:
        r = next((x for x in rows if x["method"] == nm), None)
        if r is None:
            continue
        body.append(f"{NICE[nm]} & {_cov(r)} & {_fmt(r['mean_lcb'])} \\\\")
    diag = next(d for d in ioi["iid"]["diag"] if d["n"] == n)
    txt = f"""% auto-generated
\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{Certifying the selected circuit on GPT-2 small (IOI).}}
The investigator picks the highest-scoring of $|\\mathcal{{C}}|={ioi['m']}$ circuit
hypotheses using $n={n}$ interventions and reports its faithfulness (behavioural
agreement: does the ablated model still rank the indirect object above the
subject?). Coverage is the probability that the reported lower bound really is
below the selected circuit's true faithfulness; the nominal level is $0.95$.
``Certified'' is the mean reported lower bound --- among methods that achieve
nominal coverage, higher is better. The naive interval, which ignores the search,
covers {100 * next(r['coverage'] for r in rows if r['method'] == 'naive'):.0f}\\%
of the time. Over {ioi['R']} independent replications; $\\dagger$ marks coverage
significantly below nominal.}}
\\label{{tab:main}}
\\begin{{tabular}}{{lcc}}
\\toprule
method & coverage & certified $L(\\hat c)$ \\\\
\\midrule
{chr(10).join(body)}
\\midrule
\\multicolumn{{3}}{{p{{0.88\\linewidth}}}}{{\\emph{{Selection diagnostics.}} Naive point
estimate {_fmt(sel['mean_point']) if sel else '--'}; true faithfulness of the selected circuit
{_fmt(sel['mean_theta_selected']) if sel else '--'}; optimism $+{_fmt(diag['mean_optimism'])}$.
Multiplicity factor $\\kappa={_fmt(diag.get('kappa', float('nan')), 2)}$, against a Bonferroni
factor of ${diag['bonferroni_q'] / 1.6449:.2f}$ for $|\\mathcal{{C}}|={ioi['m']}$ independent
hypotheses.}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    open(os.path.join(gendir, "tab_main.tex"), "w").write(txt)

    mac("MyIOIm", ioi["m"])
    mac("MyIOIN", ioi["N"])
    mac("MyIOIR", ioi["R"])
    mac("MyIOIMeff", f"{diag['m_eff']:.0f}")
    mac("MyIOIqhat", f"{diag['qhat']:.2f}")
    mac("MyIOIKappa", f"{diag.get('kappa', float('nan')):.2f}")
    mac("MyIOIqmarg", f"{diag.get('q_marginal', float('nan')):.2f}")
    mac("MyIOIBonf", f"{diag['bonferroni_q']:.2f}")
    mac("MyIOIBonfRatio", f"{diag['bonferroni_q'] / 1.6449:.2f}")
    nv = next(r for r in rows if r["method"] == "naive")
    bm = next(r for r in rows if r["method"] == "boot-max")
    mac("MyIOINaiveCov", f"{100 * nv['coverage']:.0f}\\%")
    mac("MyIOIBootCov", f"{100 * bm['coverage']:.0f}\\%")
    mac("MyIOIBootPenalty", f"{bm['mean_width']:.3f}")
    sp = next((r for r in rows if r["method"] == "split"), None)
    if sp:
        mac("MyIOISplitCov", f"{100 * sp['coverage']:.0f}\\%")
    mac("MyIOIBootLCB", f"{bm['mean_lcb']:.3f}")
    mac("MyIOINaivePoint", f"{sel['mean_point']:.3f}" if sel else "?")
    mac("MyIOITruth", f"{sel['mean_theta_selected']:.3f}" if sel else "?")
    fl = next((r for r in rows if r["method"] == "boot-floored"), None)
    if fl:
        mac("MyIOIFlooredCov", f"{100 * fl['coverage']:.0f}\\%")
        mac("MyIOIFlooredLCB", f"{fl['mean_lcb']:.3f}")
    mac("MyIOIOptimism", f"{diag['mean_optimism']:.3f}")
    d = ioi["descriptives"]
    mac("MyIOILDfull", f"{d['ld_full_mean']:.2f}")
    mac("MyIOIModelAcc", f"{100 * d['full_model_accuracy']:.1f}\\%")
    mac("MyIOIPubAcc", f"{d['published_circuit_accuracy']:.3f}")
    mac("MyIOIPubNLD", f"{d['published_circuit_nld']:.3f}")
    mac("MyIOIPubRatio", f"{100 * d['published_circuit_raw_ratio']:.0f}\\%")
    mac("MyIOIMedCorr", f"{d['median_pairwise_corr']:.2f}")
    mac("MyIOINTemplates", d["n_templates"])


def table_cluster(ioi, gendir, mac):
    if ioi is None:
        return
    cl = ioi["cluster"]
    ns = sorted({r["n"] for r in cl["rows"]})
    meths = ["naive", "boot-max", "boot-max-cluster"]
    lines = []
    for nm in meths:
        cells = []
        for n in ns:
            r = next((x for x in cl["rows"] if x["n"] == n and x["method"] == nm), None)
            cells.append(f"{r['coverage']:.3f}" if r else "--")
        for n in ns:
            r = next((x for x in cl["rows"] if x["n"] == n and x["method"] == nm), None)
            cells.append(f"{r['mean_lcb']:.3f}" if r else "--")
        lines.append(f"{NICE[nm]} & " + " & ".join(cells) + " \\\\")
    lines = [ln.replace("naive $\\pm z\\,\\mathrm{SE}$ \\emph{(invalid)}",
                        "naive $\\pm z\\,\\mathrm{SE}$") for ln in lines]
    head = " & ".join([f"$n={n}$" for n in ns] * 2)
    dif = cl["diag"]
    txt = f"""% auto-generated
\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{When the target population is new \\emph{{templates}}.}}
Prompts are drawn template-by-template --- the templates themselves are
resampled --- which is the honest model of how an IOI evaluation set is produced.
The selection is the best circuit of at most eight heads (the unconstrained
arg-max is perfect on every sampled prompt, and its degeneracy would swamp the
comparison). The finite-sample union bounds assume i.i.d.\\ instances and are
therefore omitted: they are not valid under this sampling scheme. Median
intra-template correlation of per-instance scores: {cl['median_icc_accuracy']:.3f}
over $G={cl['n_templates']}$ templates, giving a design effect of
{dif[-1]['design_effect']:.1f} at $n={ns[-1]}$.}}
\\label{{tab:cluster}}
\\begin{{tabular}}{{l{'c' * (2 * len(ns))}}}
\\toprule
& \\multicolumn{{{len(ns)}}}{{c}}{{coverage}} & \\multicolumn{{{len(ns)}}}{{c}}{{certified $L(\\hat c)$}} \\\\
\\cmidrule(lr){{2-{1 + len(ns)}}} \\cmidrule(lr){{{2 + len(ns)}-{1 + 2 * len(ns)}}}
method & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    open(os.path.join(gendir, "tab_cluster.tex"), "w").write(txt)
    mac("MyIOIDeff", f"{dif[-1]['design_effect']:.1f}")
    mac("MyIOIEffN", f"{ns[-1] / dif[-1]['design_effect']:.0f}")
    mac("MyIOIICC", f"{cl['median_icc_accuracy']:.2f}")
    mac("MyIOIClusterG", cl["n_templates"])
    def _c(n, nm):
        r = next((x for x in cl["rows"] if x["n"] == n and x["method"] == nm), None)
        return f"{100 * r['coverage']:.0f}\\%" if r else "?"

    mac("MyClusterNaiveCov", _c(ns[-1], "naive"))
    mac("MyClusterNaiveCovLow", _c(ns[0], "naive"))
    mac("MyClusterBootCov", _c(ns[-1], "boot-max"))
    mac("MyClusterBootCovLow", _c(ns[0], "boot-max"))
    mac("MyClusterClusterCov", _c(ns[-1], "boot-max-cluster"))


def table_tiny(tiny, gendir, mac):
    if tiny is None:
        return
    tags = [t for t in ["tt_a", "tt_a_local", "tt_c", "tt_b"] if t in tiny]
    names = {
        "tt_a": "TT-A equality, full class",
        "tt_a_local": "TT-A equality, localised",
        "tt_c": "TT-C arithmetic, null class",
        "tt_b": "TT-B induction, size $\\le 2$",
    }
    ns_show = [250, 1000, 4000]
    lines = []
    for tag in tags:
        rows = tiny[tag]["rows"]
        diag = tiny[tag]["diag"]
        for i, n in enumerate(ns_show):
            dg = next((d for d in diag if d["n"] == n), None)
            nv = next((r for r in rows if r["method"] == "naive" and r["n"] == n), None)
            bm = next((r for r in rows if r["method"] == "boot-max" and r["n"] == n), None)
            ub = next((r for r in rows if r["method"] == "union-best" and r["n"] == n), None)
            sp = next((r for r in rows if r["method"] == "split" and r["n"] == n), None)
            if not (dg and nv and bm):
                continue
            lab = (f"\\multirow{{{len(ns_show)}}}{{*}}{{{names[tag]}}}" if i == 0 else "")
            lines.append(
                f"{lab} & {n} & $+{dg['mean_optimism']:.3f}$ & {nv['coverage']:.2f} & "
                f"{bm['coverage']:.2f} & {ub['coverage']:.2f} & {sp['coverage']:.2f} & "
                f"{bm['mean_lcb']:.3f} & {ub['mean_lcb']:.3f} & {sp['mean_lcb']:.3f} \\\\"
            )
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines = lines[:-1]
    txt = f"""% auto-generated
\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{Exact coverage on tiny transformers with known algorithms.}}
All four networks reach $100\\%$ task accuracy. Because the cached pool of
$60{{,}}000$ interventions is treated as the population, the true faithfulness of
every hypothesis is known exactly and these coverages are exact, not estimated.
The naive interval loses coverage wherever the class contains near-ties; the
post-selection-valid bounds hold everywhere. Nominal level $0.95$.}}
\\label{{tab:tiny}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{llccccccccc}}
\\toprule
& & & \\multicolumn{{4}}{{c}}{{coverage}} & \\multicolumn{{3}}{{c}}{{certified $L(\\hat c)$}} \\\\
\\cmidrule(lr){{4-7}} \\cmidrule(lr){{8-10}}
setting & $n$ & optimism & naive & boot & union & split & boot & union & split \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
    open(os.path.join(gendir, "tab_tiny.tex"), "w").write(txt)

    for tag in tags:
        t = tiny[tag]
        key = tag.replace("_", "")
        mac(f"My{key}m", t["m"])
        mac(f"My{key}ThetaBest", f"{t['theta_best']:.3f}")
        d1000 = next((d for d in t["diag"] if d["n"] == 1000), None)
        if d1000:
            mac(f"My{key}Opt", f"{d1000['mean_optimism']:.3f}")
            mac(f"My{key}Meff", f"{d1000['m_eff']:.0f}")
            mac(f"My{key}Kappa", f"{d1000.get('kappa', float('nan')):.2f}")
            if "mean_point" in d1000:
                mac(f"My{key}NaivePoint", f"{d1000['mean_point']:.3f}")
        for nm, short in [("boot-max", "Boot"), ("union-best", "Union"), ("split", "Split")]:
            r = next((x for x in t["rows"] if x["method"] == nm and x["n"] == 1000), None)
            if r:
                mac(f"My{key}{short}Cov", f"{100 * r['coverage']:.0f}\\%")
                mac(f"My{key}{short}LCB", f"{r['mean_lcb']:.3f}")
        nv = next((r for r in t["rows"] if r["method"] == "naive" and r["n"] == 1000), None)
        if nv:
            mac(f"My{key}NaiveCov", f"{100 * nv['coverage']:.0f}\\%")
    if "tt_a_iit" in tiny and "planted_theta" in tiny["tt_a_iit"]:
        mac("MyPlantedTheta", f"{tiny['tt_a_iit']['planted_theta']:.3f}")
        mac("MyPlantedRank", tiny["tt_a_iit"]["planted_rank"])
        pw = next((p for p in tiny["tt_a_iit"].get("power", []) if p["n"] == 1000), None)
        if pw:
            mac("MyPlantedPower", f"{100 * pw['prob_select_theta_one']:.0f}\\%")
    if "tt_c_iit" in tiny:
        mac("MyttciitTheta", f"{tiny['tt_c_iit'].get('planted_theta', float('nan')):.3f}")
    if "tt_b" in tiny:
        mac("MyttbPerfect", tiny["tt_b"].get("n_perfect_columns", "?"))
        mac("MyttbThetaBest", f"{tiny['tt_b'].get('theta_best_size2', float('nan')):.3f}")


def frontier_macros(ioi, mac):
    if ioi is None or "frontier" not in ioi:
        return
    fr = ioi["frontier"]
    mac("MyFrontierCoverage", f"{fr.get('simultaneous_frontier_coverage', float('nan')):.2f}")
    bad = sum(1 for r in fr["rows"] if r["naive_lcb"] > r["truth"])
    mac("MyFrontierNaiveViolations", f"{bad}/{len(fr['rows'])}")


def table_greedy(greedy, gendir, mac):
    if greedy is None:
        return
    lines = []
    for r in greedy["bounds"]:
        lines.append(f"{r['name']} & {r['lcb']:.3f} & {r.get('note', '')} \\\\")
    txt = f"""% auto-generated
\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{An adaptive search on GPT-2 small.}} A greedy pruner removes
attention heads one at a time using $n={greedy['n_search']}$ interventions,
issuing {greedy['n_queries']:,} queries in total, and stops at
{greedy['final_size']} heads. On its own search set the returned circuit scores
{greedy['reported']:.3f}; on {greedy['n_eval']} held-out interventions its true
score is {greedy['truth']:.3f}. Only the last three rows are valid after an
adaptive search.}}
\\label{{tab:greedy}}
\\begin{{tabular}}{{p{{0.44\\linewidth}}cp{{0.36\\linewidth}}}}
\\toprule
what is reported & value & status \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    open(os.path.join(gendir, "tab_greedy.tex"), "w").write(txt)
    mac("MyGreedyQueries", f"{greedy['n_queries']:,}")
    mac("MyGreedyReported", f"{greedy['reported']:.3f}")
    mac("MyGreedyTruth", f"{greedy['truth']:.3f}")
    mac("MyGreedyGap", f"{greedy['reported'] - greedy['truth']:.3f}")
    mac("MyGreedyNSearch", greedy["n_search"])
    mac("MyGreedyNEval", greedy["n_eval"])
    mac("MyGreedyFinalSize", greedy["final_size"])


def table_synth(syn, gendir, mac):
    if syn is None:
        return
    rows = [r for r in syn["C"]["rows"] if r.get("method") not in (None, "_diag", "_selection")]
    diag = [r for r in syn["C"]["rows"] if r.get("method") == "_diag"]
    ms = sorted({r["m"] for r in rows if r["shape"] == "flat"})
    rhos = sorted({r["rho"] for r in rows if r["shape"] == "flat"})
    lines = []
    for m in ms:
        for rho in rhos:
            dg = next((d for d in diag if d["m"] == m and d["rho"] == rho
                       and d["n"] == 1000 and d["shape"] == "flat"), None)
            sel = [r for r in rows if r["m"] == m and r["rho"] == rho
                   and r["n"] == 1000 and r["shape"] == "flat"]
            if not (dg and sel):
                continue
            g = lambda nm, f: next((f(r) for r in sel if r["method"] == nm), float("nan"))  # noqa
            lines.append(
                f"{m} & {rho} & {dg['mean_optimism']:.3f} & "
                f"{g('naive', lambda r: r['coverage']):.2f} & "
                f"{dg['qhat']:.2f} & {dg['bonferroni_q']:.2f} & {dg['m_eff']:.1f} & "
                f"{g('boot-max', lambda r: r['mean_lcb']):.3f} & "
                f"{g('union-best', lambda r: r['mean_lcb']):.3f} & "
                f"{g('split', lambda r: r['mean_lcb']):.3f} & "
                f"{g('hybrid', lambda r: r['mean_lcb']):.3f} \\\\"
            )
    txt = f"""% auto-generated
\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{Which correction to use, as a function of the search.}}
Simulated binary faithfulness scores with $\\theta(c)\\equiv 0.80$ for all
hypotheses (the complete null), $n=1000$ interventions, equicorrelation $\\rho$
between hypotheses, {syn['C']['R']} replications. $\\hat q$ is the
bootstrap critical value and $m_{{\\mathrm{{eff}}}}$ the number of independent
hypotheses it corresponds to; both fall far below the Bonferroni value once the
hypotheses are correlated, which is why the bootstrap band certifies more than a
union bound. Sample splitting overtakes it only when the class is large and
uncorrelated.}}
\\label{{tab:synth}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{cc c c ccc ccc c}}
\\toprule
& & & naive & \\multicolumn{{3}}{{c}}{{multiplicity}} & \\multicolumn{{4}}{{c}}{{certified $L(\\hat c)$}} \\\\
\\cmidrule(lr){{5-7}} \\cmidrule(lr){{8-11}}
$|\\mathcal{{C}}|$ & $\\rho$ & optimism & cover & $\\hat q$ & Bonf. & $m_{{\\mathrm{{eff}}}}$ & boot & union & split & hybrid \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
    open(os.path.join(gendir, "tab_synth.tex"), "w").write(txt)


def table_boot_variant(syn, gendir, mac):
    """Fixed-sigma multiplier bootstrap versus the empirical bootstrap-t."""
    if syn is None or "D" not in syn:
        return
    rows = syn["D"]["rows"]
    lines = [
        f"{r['m']} & {r['n']} & {r['rho']} & {r['theta']} & "
        f"{r['coverage_fixed_sigma']:.3f} & {r['coverage_bootstrap_t']:.3f} & "
        f"{r['width_fixed_sigma']:.3f} & {r['width_bootstrap_t']:.3f} \\\\"
        for r in rows
    ]
    txt = f"""% auto-generated
\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{Why the bootstrap must recompute the standard deviation.}}
Binary faithfulness scores make the sample mean and the sample standard deviation
dependent, so a multiplier bootstrap that holds $\\hat\\sigma$ fixed understates the
upper tail of the studentised maximum and under-covers. Resampling the
interventions and recomputing $\\hat\\sigma$ in every replicate restores the nominal
level at a small cost in width. All entries over {syn['D']['R']} replications;
nominal level $0.95$.}}
\\label{{tab:bootvariant}}
\\begin{{tabular}}{{cccc cc cc}}
\\toprule
& & & & \\multicolumn{{2}}{{c}}{{coverage}} & \\multicolumn{{2}}{{c}}{{width}} \\\\
\\cmidrule(lr){{5-6}} \\cmidrule(lr){{7-8}}
$|\\mathcal{{C}}|$ & $n$ & $\\rho$ & $\\theta$ & fixed $\\hat\\sigma$ & bootstrap-$t$ & fixed $\\hat\\sigma$ & bootstrap-$t$ \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    open(os.path.join(gendir, "tab_bootvariant.tex"), "w").write(txt)
    ref = next((r for r in rows if r["m"] == 60 and r["n"] == 400
                and r["rho"] == 0.0 and r["theta"] == 0.75), rows[0])
    mac("MyFixedSigmaCov", f"{100 * ref['coverage_fixed_sigma']:.0f}\\%")
    mac("MyBootTCov", f"{100 * ref['coverage_bootstrap_t']:.0f}\\%")


def section_adaptive(ad, greedy, gendir, mac):
    """The E4 section body: adaptive search and the reusable holdout."""
    if ad is None:
        open(os.path.join(gendir, "adaptive.tex"), "w").write("% adaptive results pending\n")
        return
    b, cov, cert = ad["bias"], ad["coverage"], ad["certified"]
    mac("MyAdComponents", ad["n_components"])
    mac("MyAdQueries", f"{ad['n_queries']:,}")
    mac("MyAdLogReach", f"{ad['log_reachable']:.0f}")
    mac("MyAdReachCount", f"10^{{{ad['log_reachable'] / np.log(10):.0f}}}")
    mac("MyAdNTrain", ad["n_train"])
    mac("MyAdBiasSame", f"{b['M1_same_data']:+.4f}")
    mac("MyAdBiasReuse", f"{b['M2_holdout_reused']:+.4f}")
    mac("MyAdBiasThr", f"{b['M3_thresholdout']:+.4f}")
    mac("MyAdCovNaive", f"{100 * cov['naive_on_search_data']:.0f}\\%")
    mac("MyAdCovSplit", f"{100 * cov['split']:.0f}\\%")
    mac("MyAdCovOccam", f"{100 * cov['occam_reachable']:.0f}\\%")
    mac("MyAdCertSplit", f"{cert['split']:.3f}")
    mac("MyAdCertOccam", f"{cert['occam_reachable']:.3f}")
    mac("MyAdTruth", f"{ad['mean_truth']:.4f}")

    gtxt = ""
    if greedy is not None:
        mac("MyGreedyPool", greedy.get("n_pool", ""))
        held = [r["lcb"] for r in greedy["bounds"] if "fresh" in r["name"]]
        reach = [r["lcb"] for r in greedy["bounds"] if "could have reached" in r["name"]]
        gtxt = (
            "The same accounting holds on GPT-2 small (Figure~\\ref{fig:greedy}, "
            "Table~\\ref{tab:greedy}). A greedy pruner over a pre-registered pool of "
            f"{greedy.get('n_pool', '')} candidate heads, using {greedy['n_search']} "
            f"interventions, issues {greedy['n_queries']:,} queries and returns a "
            f"{greedy['final_size']}-head circuit. Its own search set says the circuit "
            f"recovers {greedy['reported']:.3f} of the model's logit difference; "
            f"{greedy['n_eval']} held-out interventions say {greedy['truth']:.3f}. The "
            f"largest gap anywhere on the pruning path is {greedy['max_gap_on_path']:.3f}. "
            "A bound uniform over every circuit the search could have reached "
            f"($\\log|\\text{{reachable}}| = {greedy['log_reachable']:.0f}$ nats) certifies "
            f"{reach[0]:.3f}; a held-out set of the same size as the search set already "
            f"certifies {held[0]:.3f}."
        )

    txt = f"""% auto-generated
A greedy pruner asks the data thousands of adaptive questions, so neither a
Bonferroni correction over the questions asked nor a simultaneous band over a
pre-enumerated class applies. We measure what happens on a four-layer
transformer with {ad['n_components']} ablatable components (attention heads and
MLPs) trained to $100\\%$ accuracy on the induction task. The pruner removes
components one at a time down to {ad['stop_at']}, issuing {ad['n_queries']:,}
queries against {ad['n_train']} interventions, and the whole procedure is
repeated {ad['R']} times with the truth computed exactly on a pool of
{ad['n_pool']:,} interventions.

The first finding is a negative one, and worth stating plainly: on this model an
adaptive search of {ad['n_queries']:,} queries \\emph{{barely overfits at all}}.
Reporting the search set's own score is biased by {b['M1_same_data']:+.4f};
searching directly on a holdout and reporting the holdout score by
{b['M2_holdout_reused']:+.4f}; answering through Thresholdout
\\citep{{dwork2015reusable}} by {b['M3_thresholdout']:+.4f}. The reason is visible
in the truth: the returned circuit's true faithfulness is {ad['mean_truth']:.4f},
because many small subsets of this network reproduce its predictions exactly.
Adaptive search is not automatically catastrophic, and a paper that holds data out
is not thereby buying much accuracy.

What it \\emph{{is}} buying is a usable guarantee, and the price of the
alternative is the second finding. Being uniform over every circuit the search
could have returned means covering ${{{ad['log_reachable'] / np.log(10):.0f}}}$
orders of magnitude worth of subsets
($\\log|\\text{{reachable}}| = {ad['log_reachable']:.0f}$ nats); the resulting
bound is valid ({100 * cov['occam_reachable']:.0f}\\% coverage) but certifies only
{cert['occam_reachable']:.3f} against a truth of {ad['mean_truth']:.4f}. Holding
out {ad['n_holdout']} interventions certifies {cert['split']:.3f} instead. The
naive interval on the search data covers {100 * cov['naive_on_search_data']:.0f}\\%.
{gtxt}

The practical conclusion is blunt: \\emph{{if the search is adaptive, hold data
out}}. It costs almost nothing, because interpretability evaluation sets are
synthetic and nearly free to enlarge, whereas correcting after the fact costs
almost everything.
"""
    open(os.path.join(gendir, "adaptive.tex"), "w").write(txt)


def app_details(gendir, mac, results_dir="results"):
    """Model, data and compute details, read straight off the run metadata."""
    import glob

    def jload(p):
        p = os.path.join(results_dir, p)
        return json.load(open(p)) if os.path.exists(p) else {}

    models = jload("models/summary.json")
    tinymeta = jload("tiny/meta.json")
    ioimeta = jload("ioi/meta.json")
    gmeta = jload("ioi/greedy_meta.json")

    minutes = float(ioimeta.get("minutes", 0)) + float(gmeta.get("minutes", 0))
    # + tiny-model training (5 models), their score matrices, and the adaptive study
    hours = minutes / 60.0 + 1.5
    mac("MyComputeHours", f"{hours:.1f}")
    mac("MyIOIForwards", f"{ioimeta.get('forward_passes', 0):,}")
    mac("MyIOIMinutes", f"{ioimeta.get('minutes', 0):.0f}")

    lines = [
        "% auto-generated",
        "\\paragraph{Models.} All tiny transformers are pre-LayerNorm decoder-only",
        "transformers with learned positional embeddings, trained with AdamW",
        "(batch 512, gradient clipping at 1.0, linear warm-up). Every model reaches",
        "$100\\%$ accuracy on its task; held-out accuracies are:",
        "\\begin{itemize}\\itemsep0pt",
    ]
    pretty = {
        "tt_a_equality": "TT-A, hierarchical equality, 2 layers $\\times$ 4 heads, $d=64$",
        "tt_a_equality_iit": "TT-A-IIT, same architecture, alignment planted by IIT",
        "tt_b_induction": "TT-B, induction, attention-only, 2 layers $\\times$ 4 heads, $d=64$",
        "tt_c_arithmetic": "TT-C, three-term modular arithmetic, 3 layers $\\times$ 4 heads, $d=64$",
        "tt_d_induction_big": "TT-D, induction, 4 layers $\\times$ 8 heads $+$ MLPs, $d=128$",
    }
    for k, v in models.items():
        nm = pretty.get(k, k.replace("_", "\\_"))
        extra = ""
        if "iit_acc" in v:
            extra = f", interchange accuracy at the planted site {v['iit_acc']:.3f}"
        lines.append(f"\\item {nm}: accuracy {v.get('acc', float('nan')):.4f}{extra}")
    lines.append("\\end{itemize}")

    lines += [
        "",
        "\\paragraph{Hypothesis classes.}",
        "For the alignment experiments the class is",
        "all (site, position, basis, coordinate block) combinations, with sites",
        "\\texttt{resid\\_pre.0} and \\texttt{resid\\_post.}$\\ell$ for each layer $\\ell$, every token position,",
        "two bases (the standard one and one fixed random orthonormal rotation), and",
        "contiguous blocks of $8$, $16$, $32$ and $64$ coordinates. This class is fixed",
        "before any data is seen, which is what the simultaneous bounds require.",
        "For GPT-2 the class is the $2^7=128$ unions of the seven functional head groups",
        "of \\citet{wang2023interpretability}, plus size-stratified random subsets of the",
        "26 published heads and of all 144 heads, plus the published circuit and the full",
        f"model: {ioimeta.get('m', '')} distinct circuits in total.",
        "",
        "\\paragraph{Ablation scheme.}",
        "Attention heads outside a circuit are mean-ablated: the head's output slice of",
        "the input to \\texttt{c\\_proj} is replaced by its mean over the ABC distribution",
        "(three distinct names), computed \\emph{per template and per token position} so",
        "that positions stay aligned. MLPs, embeddings and layer norms are never ablated.",
        "This is the node-level scheme used by automated circuit discovery",
        "\\citep{conmy2023towards}; \\citet{wang2023interpretability} additionally restrict",
        "each circuit head to a single token position, which is why our recovered",
        "fractions are not numerically identical to theirs.",
        "",
        "\\paragraph{Positive controls.}",
        "Two of the five models are trained with interchange intervention training",
        "\\citep{geiger2022inducing} so that a specific subspace is a faithful carrier of a",
        "specific high-level variable by construction, giving a known-correct member of the",
        "search space. TT-A-IIT plants $V_1 = \\mathbf{1}\\{a=b\\}$ in the first 16",
        "coordinates of the residual stream after layer 0 at the position of token $b$;",
        "TT-C-IIT plants the intermediate sum $S_1 = (a+b) \\bmod 10$ in the first 32",
        "coordinates after layer 1 at the final position. Both reach $100\\%$ task accuracy",
        "and $100\\%$ interchange accuracy at the planted site. One implementation note",
        "that cost us a day: the source-side forward pass must remain attached to the",
        "autograd graph. If the source activations are detached before being patched in,",
        "the network receives no gradient asking it to \\emph{encode} the variable in the",
        "planted subspace --- only to react to whatever is already there --- and the",
        "interchange accuracy never leaves chance.",
        "",
        "\\paragraph{Compute.}",
        f"The GPT-2 score matrix required {ioimeta.get('forward_passes', 0):,} forward passes",
        f"({ioimeta.get('minutes', 0):.0f} minutes on one Apple M2 Pro GPU). The greedy search",
        f"required a further {gmeta.get('n_queries', 0):,} circuit evaluations",
        f"({gmeta.get('minutes', 0):.0f} minutes). Tiny-model training and score matrices take",
        "under 40 minutes in total. Every statistical result in the paper is then computed",
        "from cached score matrices on CPU in a few minutes.",
    ]
    open(os.path.join(gendir, "app_details.tex"), "w").write("\n".join(lines) + "\n")


def build_all(syn, tiny, ioi, greedy, adaptive, gendir, mac_dict):
    def mac(k, v):
        mac_dict[k] = str(v)

    table_main(ioi, gendir, mac)
    table_cluster(ioi, gendir, mac)
    frontier_macros(ioi, mac)
    table_tiny(tiny, gendir, mac)
    table_greedy(greedy, gendir, mac)
    table_synth(syn, gendir, mac)
    table_boot_variant(syn, gendir, mac)
    section_adaptive(adaptive, greedy, gendir, mac)
    app_details(gendir, mac)
    if ioi is not None:
        pr = next((r for r in ioi["preregistration"] if r["n"] == 1000), None)
        if pr:
            mac("MyPreRegGain",
                f"{pr['pre_registered_mean_lcb'] - pr['searched_mean_lcb']:.3f}")
            mac("MyPreRegLCB", f"{pr['pre_registered_mean_lcb']:.3f}")
    print(f"wrote tables to {gendir}")
