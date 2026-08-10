# Theory notes (working document for the paper)

## 1. Setup

An **evaluation instance** `Z` is whatever randomness a single faithfulness
measurement consumes.  For an interchange-intervention experiment in causal
abstraction, `Z = (b, s)` is a pair (base input, source input); for a circuit
ablation experiment, `Z` is a single input.  We assume `Z_1, ..., Z_n ~ iid P`.

A **circuit hypothesis** `c ∈ C` is any object that, together with `Z`, defines
a *bounded per-instance score*

    φ(c; Z) ∈ [0, 1].

Examples.
* interchange-intervention accuracy (IIA): `φ = 1{ f_L(b ← Π(V)(s)) = f_H(b ← V(s)) }`;
* behavioural agreement under ablation: `φ = 1{ argmax under c equals argmax of the full model }`;
* causal-scrubbing style agreement rates.

**Graded faithfulness** is the population mean

    θ(c) = E_P[ φ(c; Z) ],      θ̂_n(c) = (1/n) Σ_i φ(c; Z_i).

**Selection.**  A search procedure `A` maps the data to a hypothesis
`ĉ = A(Z_1..Z_n) ∈ C`.  The canonical case is `ĉ = argmax_c θ̂_n(c)`.  Almost all
published circuit analyses report `θ̂_n(ĉ)` computed on the same interventions
used by `A`.

**Selection optimism.**  `Δ_n(A) = E[ θ̂_n(ĉ) − θ(ĉ) ]`.

Note the estimand: we ask how faithful *the reported circuit* is, not how
faithful the best possible circuit is.  This is the quantity a reader of an
interpretability paper cares about.

## 2. How bad is the naive report?

### Prop. 1 (upper bound on optimism)
For any selection rule `A`,
`Δ_n(A) ≤ E sup_{c∈C} (θ̂_n(c) − θ(c))`,
and by Bernstein's inequality plus a union bound over `|C| = m`,

    E sup_c (θ̂_n − θ) ≤ sqrt( 2 σ²_max log m / n ) + 2 log m / (3n),
    σ²_max = max_c Var φ(c; Z).

*Proof.* Bernstein gives `P(θ̂(c) − θ(c) > t) ≤ exp(−n t²/(2σ² + 2t/3))`.  Union
over `m`, then `E X ≤ u + ∫_u^∞ P(X>t) dt` with `u` the solution of
`m exp(−nu²/(2σ²+2u/3)) = 1`. ∎

### Prop. 2 (the rate is not an artefact of the bound)
If the centred scores are iid `N(0, σ²)` across instances *and independent
across hypotheses*, and `θ(c) = θ₀` for all `c`, then

    Δ_n = (σ/√n) E[max_{j≤m} N_j]  and  c₀ σ sqrt(log m / n) ≤ Δ_n ≤ σ sqrt(2 log m / n)

with `c₀ = 1/√(π log 2) ≈ 0.6797` for `m ≥ 2`
(Boucheron, Lugosi & Massart 2013, §2.5).  So the `sqrt(log m / n)` rate is
exact up to a constant.  For bounded scores the same order holds by the
Gaussian approximation for maxima (Chernozhukov, Chetverikov & Kato 2013).

### Prop. 3 (exact coverage of the naive interval)
Let `G = (√n (θ̂(c) − θ(c))/σ(c))_c` and suppose `θ(c) = θ₀` for all `c` (the
complete null: the class contains no genuinely better hypothesis).  The naive
one-sided interval at the arg-max covers iff `max_c G_c ≤ z_{1−α}`.  Under an
equicorrelated Gaussian limit with correlation ρ,

    coverage → E_W[ Φ( (z_{1−α} − √ρ W)/√(1−ρ) )^m ],   W ~ N(0,1),

which equals `Φ(z_{1−α})^m` at ρ = 0.  Numbers at α = 0.05:

| m      | ρ=0    | ρ=0.3  | ρ=0.6  | ρ=0.9  |
|--------|--------|--------|--------|--------|
| 10     | 0.599  | 0.693  | 0.784  | 0.885  |
| 100    | 0.006  | 0.246  | 0.530  | 0.813  |
| 1000   | 0.000  | 0.041  | 0.308  | 0.742  |
| 10000  | 0.000  | 0.004  | 0.161  | 0.673  |

Two lessons: (i) the nominal 95% interval can have essentially zero coverage;
(ii) correlation between hypotheses is what saves it, and circuit hypotheses are
strongly correlated, so the honest answer is somewhere in between and must be
*measured*, not assumed.

## 3. Bounds that survive selection

Throughout, a lower confidence bound `L` is **post-selection valid at level α**
if `P(θ(ĉ) ≥ L(ĉ)) ≥ 1 − α` for every selection rule `ĉ` measurable w.r.t. the
data used to build `L`.  The cleanest sufficient condition is that `L` is a
*simultaneous* band: `P(∀c: θ(c) ≥ L(c)) ≥ 1 − α`.

### (a) Finite-sample simultaneous bound (UNION-EB)
Maurer & Pontil (2009, Thm. 4): for `X_i ∈ [0,1]` iid, w.p. `≥ 1 − δ`,

    E X ≥ X̄ − sqrt( 2 V_n log(2/δ) / n ) − 7 log(2/δ) / (3(n−1)),  V_n = sample variance.

Applying it at `δ = α/m` and taking a union gives a simultaneous band.
Distribution-free, finite-sample, valid for *any* selection rule.  Because a
faithful circuit has `θ(c)` near 1 and hence tiny variance, the empirical
Bernstein term is much smaller than Hoeffding's; because an unfaithful one has
variance near 1/4, Hoeffding can be tighter there.  Running both at `α/2` and
taking the larger bound is valid and never much worse than either.

### (b) Occam / prior-weighted bound (huge or infinite classes)
For any prior `π` on `C` fixed before seeing data, replace `α/m` by `απ(c)`:

    θ(c) ≥ θ̂(c) − sqrt(2 V_n(c) L_c / n) − 7 L_c/(3(n−1)),  L_c = log(2/(α π(c))),

simultaneously over the whole class.  With the size-stratified prior over
subsets of `N` components, `π(c) = 1/((N+1) binom(N,|c|))`, so

    L_c = log(2/α) + log(N+1) + log binom(N, |c|).

This makes precise the fact that **small circuits are statistically cheaper to
certify**: for `N = 144` (GPT-2 small's heads), `log binom(144,10) = 34.3`
whereas `log binom(144,72) = 97.1`.

### (c) Multiplier-bootstrap simultaneous band (BOOT-MAX) — the main tool
Let `e_i(c) = φ(c;Z_i) − θ̂(c)`, `σ̂(c)` the sample sd, and

    q̂ = (1−α) quantile of  max_c ( Σ_i ξ_i e_i(c) ) / (√n σ̂(c)),  ξ_i ~iid N(0,1).

Then `L(c) = θ̂(c) − q̂ σ̂(c)/√n` is an asymptotically valid simultaneous band,
uniformly over classes whose size may grow sub-exponentially in `n`
(Chernozhukov, Chetverikov & Kato 2013, 2017).  Unlike Bonferroni it *adapts to
the dependence between hypotheses*, which for nested / overlapping circuits is
strong.

**Effective number of hypotheses.**  Define `m_eff = α / (1 − Φ(q̂))`: the number
of *independent* hypotheses that would need the same critical value.  Reporting
`m_eff` alongside `|C|` tells a reader how much multiplicity the search really
incurred.

### (d) Sample splitting (SPLIT)
Select on a fraction `f` of the instances, bound on the rest with (a) at
`m = 1`.  Valid for arbitrary, even adaptive, selection.  Width `≈ z σ/√((1−f)n)`.

**When does BOOT-MAX beat splitting?**  Comparing half-widths,
`q̂ σ/√n` versus `z_{1−α} σ/√((1−f) n)`: the band is tighter iff
`q̂ ≤ z_{1−α}/√(1−f)`; at `f = 1/2` this is `q̂ ≤ √2 z_{1−α} ≈ 2.33`, i.e.
`m_eff ≲ 5`.  Splitting additionally *degrades the selection*, because `ĉ` is
chosen from `fn` instances; the fair comparison is therefore the **certified
value** `L(ĉ)` — among procedures with at least nominal coverage, higher is
better.  We use that as the headline comparison throughout.

### (e) Selective inference conditional on the arg-max (COND / HYBRID)
Under the normal approximation `θ̂ ~ N(θ, Σ/n)`, `{argmax = ĉ}` is a polyhedron
and `θ̂_ĉ` conditional on it is a truncated normal (Lee et al. 2016; Andrews,
Kitagawa & McCloskey 2024).  Inverting gives a *conditionally* valid interval.
This is powerful when the winner wins clearly and **degenerates to an
uninformative bound when several hypotheses are nearly tied** — which is the
common case in circuit search.  The hybrid of Andrews et al. (intersect with a
simultaneous `1−β` band, then condition, at level `(α−β)/(1−β)`) removes the
degeneracy at the cost of a small extra term.

### (f) Randomised selection / data fission
For a Gaussian mean vector, splitting off `U = θ̂ + τξ` for selection and
`V = θ̂ − τ^{-1}Σ̂ξ/n` for inference gives independent `U, V`.  Setting
`τ² = aσ²/n` yields `Var(U) = σ²(1+a)/n`, `Var(V) = σ²(1+1/a)/n`, exactly the
variances of an `f = 1/(1+a)` sample split.  **Randomised selection is therefore
asymptotically equivalent to splitting, not better**, unless one also *carves*
(re-uses the selection information at the inference stage), which we leave to
future work.

## 4. What is the population?  Clustered instances

Interpretability datasets are usually generated from a small number of
**templates**.  Write instance `i` as `(g_i, w_i)` with `g` the template.  Two
different estimands:

* `θ_fix` — average over new *fills* of the templates in hand;
* `θ_new` — average over new *templates* as well.

The i.i.d.-instance bootstrap targets `θ_fix`; generalising to `θ_new` requires
resampling templates.  With `G` templates, `n/G` instances each and
intra-template correlation `ρ_I` (ICC), the variance of `θ̂` is inflated by the
design effect

    DEFF = 1 + (n/G − 1) ρ_I,

so the *effective sample size* is `n / DEFF`, which saturates at `G/ρ_I`
regardless of how many prompts are generated.  Generating more prompts from the
same templates buys nothing beyond that ceiling.  We report both estimands.

For the simultaneous band, replace the i.i.d. multiplier bootstrap by the wild
*cluster* bootstrap (Rademacher multipliers at the cluster level; Cameron,
Gelbach & Miller 2008), studentised by the cluster-robust standard error, and
inflate the critical value by `t_{G−1}/z` when `G` is small.

## 5. Adaptive search

Greedy pruning issues `Q` queries, each depending on previous answers.  A union
bound over the `Q` realised queries is **not** valid because the query set is
random.  Three valid options:

1. **Split** — select adaptively on one part, certify on a held-out part.  Cost
   is only the reduced evaluation sample.  In interpretability, evaluation data
   is usually synthetic and nearly free, so this is the default recommendation.
2. **Uniformity over the reachable set** — a union (or Occam) bound over every
   circuit the search could have returned.  For greedy removal of at most `d` of
   `N` components, `log|reachable| = log Σ_{k≤d} binom(N,k)`, e.g. 99.8 nats for
   `N=144, d=124` — a Hoeffding penalty of `sqrt(99.8/(2n))`, i.e. 0.158 at
   `n = 2000`.
3. **Reusable holdout / Thresholdout** (Dwork et al. 2015) — answer adaptive
   queries from the holdout through a noisy threshold, spending budget only on
   queries that actually overfit.  Its theoretical constants are loose, so we
   report it as an empirically validated option rather than a certificate.

## 6. Metrics that are not means

The normalised logit difference recovered

    F(c) = ( E LD(c) − E LD(∅) ) / ( E LD(full) − E LD(∅) )

is a smooth functional of a mean vector.  Its influence function is

    ψ_i(c) = [ (LD_i(c) − a_c) − (LD_i(∅) − b) ] / (f − b)
             − F(c) [ (LD_i(full) − f) − (LD_i(∅) − b) ] / (f − b),

and `F̂(c) − F(c) = mean_i ψ_i(c) + o_P(n^{-1/2})`.  Feeding `ψ` to the same
multiplier bootstrap gives a simultaneous band for the whole family.  Note that
`F` is *not* an average of bounded per-instance quantities, so the finite-sample
bounds of §3(a,b) do not apply directly; we give a conservative Fieller-style
alternative in the appendix.
