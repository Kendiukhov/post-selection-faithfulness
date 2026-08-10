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
        body.append(
            f"{NICE[nm]} & {_cov(r)} & {_fmt(r['mean_lcb'])} & {_fmt(r['mean_width'])} \\\\"
        )
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
\\begin{{tabular}}{{lccc}}
\\toprule
method & coverage & certified $L(\\hat c)$ & width $\\hat\\theta(\\hat c)-L(\\hat c)$ \\\\
\\midrule
{chr(10).join(body)}
\\midrule
\\multicolumn{{4}}{{l}}{{\\emph{{selection diagnostics:}} naive point estimate
{_fmt(sel['mean_point']) if sel else '--'}, truth {_fmt(sel['mean_theta_selected']) if sel else '--'},
optimism $+{_fmt(diag['mean_optimism'])}$; multiplicity factor
$\\kappa={_fmt(diag.get('kappa', float('nan')), 2)}$, i.e.\\
$m_{{\\mathrm{{eff}}}}={diag['m_eff']:.0f}$ effective hypotheses out of $|\\mathcal{{C}}|={ioi['m']}$}} \\\\
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
    nv = next(r for r in rows if r["method"] == "naive")
    bm = next(r for r in rows if r["method"] == "boot-max")
    mac("MyIOINaiveCov", f"{100 * nv['coverage']:.0f}\\%")
    mac("MyIOIBootCov", f"{100 * bm['coverage']:.0f}\\%")
    mac("MyIOIBootPenalty", f"{bm['mean_width']:.3f}")
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
    meths = ["naive", "boot-max", "union-best", "split", "boot-max-cluster"]
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
    head = " & ".join([f"$n={n}$" for n in ns] * 2)
    dif = cl["diag"]
    txt = f"""% auto-generated
\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{When the target population is new \\emph{{templates}}.}}
Prompts are drawn template-by-template (the templates themselves are resampled),
which is the honest model of how an IOI evaluation set is produced. Bounds that
treat prompts as independent under-cover badly; the cluster bootstrap, which
resamples templates, restores validity at the price of a wider interval. Median
intra-template correlation of per-instance scores: {cl['median_icc_accuracy']:.3f}
over $G={cl['n_templates']}$ templates, giving a design effect of
{dif[-1]['design_effect']:.0f} at $n={ns[-1]}$.}}
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
    mac("MyIOIDeff", f"{dif[-1]['design_effect']:.0f}")
    mac("MyIOIICC", f"{cl['median_icc_accuracy']:.2f}")
    mac("MyIOIClusterG", cl["n_templates"])
    nvc = next(x for x in cl["rows"] if x["n"] == ns[-1] and x["method"] == "naive")
    bmc = next(x for x in cl["rows"] if x["n"] == ns[-1] and x["method"] == "boot-max")
    ccc = next(x for x in cl["rows"] if x["n"] == ns[-1] and x["method"] == "boot-max-cluster")
    mac("MyClusterNaiveCov", f"{100 * nvc['coverage']:.0f}\\%")
    mac("MyClusterBootCov", f"{100 * bmc['coverage']:.0f}\\%")
    mac("MyClusterClusterCov", f"{100 * ccc['coverage']:.0f}\\%")


def table_tiny(tiny, gendir, mac):
    if tiny is None:
        return
    tags = [t for t in ["tt_a", "tt_a_local", "tt_a_iit", "tt_c", "tt_b"] if t in tiny]
    names = {
        "tt_a": "TT-A equality, full class",
        "tt_a_local": "TT-A equality, localised",
        "tt_a_iit": "TT-A-IIT, planted",
        "tt_c": "TT-C arithmetic, null class",
        "tt_b": "TT-B induction, head subsets",
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
\\begin{{tabular}}{{llccccccccc}}
\\toprule
& & & \\multicolumn{{4}}{{c}}{{coverage}} & \\multicolumn{{3}}{{c}}{{certified $L(\\hat c)$}} \\\\
\\cmidrule(lr){{4-7}} \\cmidrule(lr){{8-10}}
setting & $n$ & optimism & naive & boot & union & split & boot & union & split \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
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
        nv = next((r for r in t["rows"] if r["method"] == "naive" and r["n"] == 1000), None)
        if nv:
            mac(f"My{key}NaiveCov", f"{100 * nv['coverage']:.0f}\\%")
    if "tt_a_iit" in tiny and "planted_theta" in tiny["tt_a_iit"]:
        mac("MyPlantedTheta", f"{tiny['tt_a_iit']['planted_theta']:.3f}")
        mac("MyPlantedRank", tiny["tt_a_iit"]["planted_rank"])


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
\\begin{{tabular}}{{lcl}}
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
\\begin{{tabular}}{{cc c c ccc ccc c}}
\\toprule
& & & naive & \\multicolumn{{3}}{{c}}{{multiplicity}} & \\multicolumn{{4}}{{c}}{{certified $L(\\hat c)$}} \\\\
\\cmidrule(lr){{5-7}} \\cmidrule(lr){{8-11}}
$|\\mathcal{{C}}|$ & $\\rho$ & optimism & cover & $\\hat q$ & Bonf. & $m_{{\\mathrm{{eff}}}}$ & boot & union & split & hybrid \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
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
    mac("MyAdNTrain", ad["n_train"])
    mac("MyAdBiasSame", f"{b['M1_same_data']:+.3f}")
    mac("MyAdBiasReuse", f"{b['M2_holdout_reused']:+.3f}")
    mac("MyAdBiasThr", f"{b['M3_thresholdout']:+.3f}")
    mac("MyAdBiasFresh", f"{b['M4_fresh_holdout']:+.3f}")
    mac("MyAdCovNaive", f"{100 * cov['naive_on_search_data']:.0f}\\%")
    mac("MyAdCovSplit", f"{100 * cov['split']:.0f}\\%")
    mac("MyAdCovOccam", f"{100 * cov['occam_reachable']:.0f}\\%")
    mac("MyAdCertSplit", f"{cert['split']:.3f}")
    mac("MyAdCertOccam", f"{cert['occam_reachable']:.3f}")
    mac("MyAdTruth", f"{ad['mean_truth']:.3f}")

    gtxt = ""
    if greedy is not None:
        mac("MyGreedyPool", greedy.get("n_pool", ""))
        gtxt = (
            f"The same picture holds on GPT-2 small. A greedy pruner over a "
            f"pre-registered pool of {greedy.get('n_pool', '')} candidate heads, using "
            f"{greedy['n_search']} interventions, issues {greedy['n_queries']:,} queries and "
            f"returns a {greedy['final_size']}-head circuit that scores "
            f"{greedy['reported']:.3f} on its own search set but only "
            f"{greedy['truth']:.3f} on {greedy['n_eval']} held-out interventions "
            f"(Figure~\\ref{{fig:greedy}} and Table~\\ref{{tab:greedy}}). A bound that is "
            f"uniform over every circuit the search could have returned "
            f"($\\log|\\text{{reachable}}| = {greedy['log_reachable']:.0f}$ nats) is valid but "
            f"vacuous at this sample size; a held-out set of the same size as the search set "
            f"already certifies "
            f"{[r['lcb'] for r in greedy['bounds'] if 'fresh' in r['name']][0]:.3f}."
        )

    txt = f"""% auto-generated
A greedy pruner asks the data thousands of adaptive questions, so neither a
Bonferroni correction over the questions asked nor a simultaneous band over a
pre-enumerated class applies. We measure what actually happens, on a
four-layer transformer with {ad['n_components']} ablatable components (attention
heads and MLPs) trained to $100\\%$ accuracy on the induction task. The pruner
removes components one at a time down to {ad['stop_at']}, issuing
{ad['n_queries']:,} queries against {ad['n_train']} interventions, and the whole
procedure is repeated {ad['R']} times with the truth computed exactly on a pool
of {ad['n_pool']:,} interventions.

Four ways of answering the pruner's queries give four very different reports of
the returned circuit's faithfulness (true value {ad['mean_truth']:.3f} on average):

\\begin{{itemize}}\\itemsep2pt
\\item \\textbf{{Report the search set's own score}} --- biased by
{b['M1_same_data']:+.3f}. The naive interval built on the search data covers only
{100 * cov['naive_on_search_data']:.0f}\\% of the time.
\\item \\textbf{{Search directly on the holdout and report the holdout score}} ---
biased by {b['M2_holdout_reused']:+.3f}. A holdout that is queried adaptively is
not a holdout.
\\item \\textbf{{Answer through Thresholdout}} \\citep{{dwork2015reusable}} --- bias
{b['M3_thresholdout']:+.3f}. The mechanism spends its budget only on queries that
actually overfit, and the holdout survives the search.
\\item \\textbf{{Search on one part, report on an untouched part}} --- bias
{b['M4_fresh_holdout']:+.3f}, coverage {100 * cov['split']:.0f}\\%, certifying
{cert['split']:.3f}.
\\end{{itemize}}

The rigorous alternative to holding data out is to be uniform over every circuit
the search \\emph{{could}} have returned. Here that set has
$\\log|\\text{{reachable}}| = {ad['log_reachable']:.0f}$ nats, and the resulting bound,
though valid ({100 * cov['occam_reachable']:.0f}\\% coverage), certifies only
{cert['occam_reachable']:.3f} --- far less than the {cert['split']:.3f} obtained by
simply holding interventions out. {gtxt}

The practical conclusion is blunt: \\emph{{if the search is adaptive, hold data
out}}. Interpretability evaluation sets are synthetic and nearly free to enlarge,
so this costs almost nothing, whereas trying to correct after the fact costs
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
    hours = minutes / 60.0 + 0.6  # tiny-model training and score matrices
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
        "$\\{(\\text{site}, \\text{position}, \\text{basis}, \\text{coordinate block})\\}$ with sites",
        "$\\{\\texttt{resid\\_pre.0}\\}\\cup\\{\\texttt{resid\\_post.}\\ell\\}$, every token position,",
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
        "\\paragraph{A failed positive control on TT-C.}",
        "We attempted to \\emph{plant} the intermediate sum $S_1$ in the arithmetic",
        "model by interchange intervention training, so that the negative result of",
        "Section~\\ref{sec:e2}(d) would have a positive control. Two planting sites were",
        "tried (the residual stream after layer 0 at the position of token $b$, and after",
        "layer 1 at the final position, with 16 and 32 dimensions respectively), with the",
        "interchange loss weighted up to twice the task loss and up to 20{,}000 steps. In",
        "both runs the task accuracy reached $100\\%$ while the interchange accuracy at the",
        "planted site stayed at chance. We report this because it bounds what the TT-C",
        "result can be read as showing: our search found no faithful single-position",
        "alignment, and we were also unable to create one.",
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
