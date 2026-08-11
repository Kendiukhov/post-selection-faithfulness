# Post-Selection-Valid Confidence Bounds for Mechanistic Faithfulness

**How faithful is the circuit you found?**

📄 **Paper:** [`paper/main.pdf`](https://github.com/Kendiukhov/post-selection-faithfulness/blob/main/paper/main.pdf) (preprint, 26 pp.) ·
[`submission/manuscript.pdf`](https://github.com/Kendiukhov/post-selection-faithfulness/blob/main/submission/manuscript.pdf) (journal version)
· archived at [`10.5281/zenodo.21891261`](https://doi.org/10.5281/zenodo.21891261)

Mechanistic interpretability reports a faithfulness score — interchange-intervention
accuracy, logit difference recovered, behavioural agreement under ablation — for a
circuit that was *selected* by maximising that same score on the same interventions.
A maximum over a search is biased upwards, and the usual `± 1.96 · SE` error bar is
not valid for it.

This repository contains `psf`, a small library that turns any per-instance
faithfulness score into a **lower confidence bound that stays valid however the
circuit was chosen**, together with the experiments and the paper.

```python
import numpy as np
from psf import bootstrap_max_lcb, effective_num_hypotheses

# S[i, c] = per-instance faithfulness score of circuit hypothesis c on intervention i
S = np.load("scores.npy")            # shape (n, m)

lcb, qhat = bootstrap_max_lcb(S, alpha=0.05, return_q=True)   # empirical bootstrap-t
chat = int(S.mean(0).argmax())       # however you picked your circuit

print(f"reported faithfulness : {S.mean(0)[chat]:.3f}")
print(f"certified (95% lower) : {lcb[chat]:.3f}")
print(f"effective #hypotheses : {effective_num_hypotheses(qhat):.0f}  (nominal {S.shape[1]})")
```

The band is *simultaneous* over the whole hypothesis class, so it is valid for a
circuit chosen by an arg-max, by greedy pruning, by eyeballing a plot — or for an
entire reported size/faithfulness frontier at once.

---

## What is in here

```
src/psf/                library (pure NumPy/SciPy for the statistics; PyTorch only for the models)
  bounds.py             every lower confidence bound: naive, Hoeffding/empirical-Bernstein
                        union, Occam (prior-weighted), empirical bootstrap-t simultaneous
                        band (+ finite-sample-floored variant), wild cluster bootstrap-t,
                        sample splitting, selective (polyhedral) and hybrid intervals,
                        betting bound
  functional.py         the same machinery for metrics that are ratios of means
                        (normalised logit difference recovered), via influence functions
  evaluate.py           replicate driver: coverage / certified value for any score matrix
  adaptive.py           greedy pruning, Thresholdout (reusable holdout), reachable-set sizes
  simulate.py           synthetic score-matrix generators with controllable correlation
  models.py             a small, fully hookable decoder-only transformer
  tasks.py              algorithmic tasks with explicit high-level causal models
  training.py           ordinary training and interchange intervention training (IIT)
  interventions.py      interchange interventions on subspaces; component ablation
  ioi.py                IOI dataset and head-level patching for HuggingFace GPT-2

experiments/            one script per experiment; each writes JSON to results/
paper/                  LaTeX source; every number is auto-generated into paper/generated/
                        (abstract.tex, body.tex and appendix.tex are shared with the
                        journal submission, so the two versions cannot drift apart)
submission/             the Neural Networks (Elsevier) package: elsarticle manuscript,
                        cover letter, highlights, declarations and an upload checklist
tests/                  unit tests, including Monte-Carlo coverage checks of every bound
scripts/run_all.sh      the full pipeline, in order
scripts/build_submission.sh  builds and checks the journal submission
```

## Reproducing the paper

```bash
pip install -e .
bash scripts/run_all.sh          # ~5 GPU-hours on one consumer accelerator
cd paper && latexmk -pdf main.tex
```

The expensive steps are the GPU passes that build **score matrices**
(`ioi_score_matrix.py`, `tiny_score_matrices.py`, `ioi_greedy.py`). Everything
statistical runs afterwards on the cached matrices and takes minutes on a CPU.
`run_all.sh` skips any stage whose output already exists.

**The cached score matrices are committed** (`results/ioi/ioi_scores.npz` and
friends), so every statistical result in the paper — every coverage number, every
certified bound, every figure — can be reproduced without a GPU:

```bash
python experiments/ioi_analysis.py --scores results/ioi/ioi_scores.npz --out results/ioi
python experiments/make_figures.py
```

The one exception is `results/tiny/tt_b_scores.npz` (57 MB), which is excluded
for size; re-run `tiny_score_matrices.py --which B` to rebuild it.

### Set `OPENBLAS_NUM_THREADS=1`

On a busy machine a multi-threaded BLAS can make the small mat-muls in the
bootstrap *a thousand times* slower than the single-threaded version, because the
thread pool spin-waits. Every script in `experiments/` pins the pools to one
thread before importing NumPy; if you call the library directly, do the same.

### Design choice that makes the numbers exact

We cache a large pool of per-instance scores and then **treat that pool as the
population**, drawing analysis samples i.i.d. from it. The true faithfulness of
every hypothesis is then the pooled column mean, known without error, so a
measured coverage carries only Monte-Carlo error over replications — it is not
contaminated by uncertainty about the truth, which is the usual obstacle to
measuring coverage on a real model.

### One implementation detail worth knowing

The simultaneous band is an **empirical bootstrap-t**: the standard deviation is
recomputed inside every resample. The textbook multiplier bootstrap, which holds
it fixed, under-covers badly for binary faithfulness scores — 58% instead of 95%
at θ = 0.95 in our simulations — because the sample mean and the sample standard
deviation of a Bernoulli variable are dependent. See
`experiments/synthetic_study.py` part D and the corresponding test.

## Which bound should I use?

| situation | use | why |
|---|---|---|
| class fixed in advance, enumerable | `bootstrap_max_lcb` | tightest; adapts to the strong correlation between overlapping circuits; certifies every hypothesis at once |
| ...and some hypotheses may be perfect on the sample | `floored_bootstrap_lcb` | the band degenerates for a zero-variance column; this uses the finite-sample bound there and the band elsewhere, each at $\alpha/2$ |
| class is a power set of components | `occam_lcb` with `size_stratified_log_prior` | handles $2^{144}$ circuits; small circuits are cheaper to certify |
| want a finite-sample, assumption-free guarantee | `union_lcb` | Hoeffding ∨ empirical Bernstein, each at $\alpha/2$ |
| search was adaptive (greedy, gradient-based, human-in-the-loop) | `split_lcb`, or hold out fresh interventions | uniformity over the *reachable* set is usually vacuous |
| instances come from a few templates and you want to generalise to new ones | `cluster_bootstrap_max_lcb` | resamples templates, not prompts |
| metric is a ratio of means (logit difference recovered) | `functional.ratio_band` | recomputes the ratio, its influence function and its standard error in every resample |

`conditional_winner_lcb` / `hybrid_winner_lcb` implement selective inference
conditional on the arg-max. They are included for completeness; the conditional
interval degenerates whenever several hypotheses are nearly tied, which is the
normal situation in circuit search.

## Tests

```bash
pytest tests -q
```

The test suite includes Monte-Carlo coverage checks: each bound is run over many
simulated datasets and its empirical coverage is asserted to be at least
nominal (and, for the naive bound, asserted to be *below* nominal — the failure
this paper is about).

## Paper

The compiled preprint is [`paper/main.pdf`](https://github.com/Kendiukhov/post-selection-faithfulness/blob/main/paper/main.pdf); its source is
in `paper/`. The version prepared for submission to *Neural Networks* (Elsevier),
together with the cover letter, highlights and author declarations, is in
[`submission/`](submission/).

**Author.** Ihor Kendiukhov, Institute of Medical Genetics and Applied Genomics, University of Tubingen, Tubingen, Germany (`ihor.kendiukhov@student.uni-tuebingen.de`).

**Archive.** This repository is archived at Zenodo:
[`10.5281/zenodo.21891261`](https://doi.org/10.5281/zenodo.21891261). Machine-readable citation metadata is in
`CITATION.cff`.

## License

MIT (see `LICENSE`). The IOI templates follow the design of Wang et al. (2023);
GPT-2 weights are downloaded from HuggingFace at run time and are not included.
