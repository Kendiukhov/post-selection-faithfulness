# Post-Selection-Valid Confidence Bounds for Mechanistic Faithfulness

**How faithful is the circuit you found?**

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

lcb, qhat = bootstrap_max_lcb(S, alpha=0.05, return_q=True)
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
                        union, Occam (prior-weighted), multiplier-bootstrap simultaneous band,
                        wild cluster bootstrap, sample splitting, selective (polyhedral) and
                        hybrid intervals, betting bound
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
tests/                  unit tests, including Monte-Carlo coverage checks of every bound
scripts/run_all.sh      the full pipeline, in order
```

## Reproducing the paper

```bash
pip install -e .
bash scripts/run_all.sh          # ~5 GPU-hours on one consumer accelerator
cd paper && latexmk -pdf main.tex
```

The expensive steps are the two GPU passes that build **score matrices**
(`ioi_score_matrix.py`, `tiny_score_matrices.py`). Everything statistical runs
afterwards on the cached matrices and takes minutes on a CPU. `run_all.sh` skips
any stage whose output already exists.

### Design choice that makes the numbers exact

Every coverage number in the paper is *exact*, not estimated: we cache a large
pool of per-instance scores and then **treat that pool as the population**,
drawing analysis samples i.i.d. from it. The true faithfulness of every
hypothesis is the pooled column mean, known without error, so "coverage" is a
Monte-Carlo estimate of a probability whose target is known exactly rather than
a comparison against a noisy oracle.

## Which bound should I use?

| situation | use | why |
|---|---|---|
| class fixed in advance, enumerable | `bootstrap_max_lcb` | tightest; adapts to the strong correlation between overlapping circuits; certifies every hypothesis at once |
| class is a power set of components | `occam_lcb` with `size_stratified_log_prior` | handles $2^{144}$ circuits; small circuits are cheaper to certify |
| want a finite-sample, assumption-free guarantee | `union_lcb` | Hoeffding ∨ empirical Bernstein, each at $\alpha/2$ |
| search was adaptive (greedy, gradient-based, human-in-the-loop) | `split_lcb`, or hold out fresh interventions | uniformity over the *reachable* set is usually vacuous |
| instances come from a few templates and you want to generalise to new ones | `cluster_bootstrap_max_lcb` | resamples templates, not prompts |
| metric is a ratio of means (logit difference recovered) | `functional.influence_band` | delta method + the same bootstrap |

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

## Citation

The paper is included in `paper/`. A `CITATION.cff` will be added on release.

## License

MIT (see `LICENSE`). The IOI templates follow the design of Wang et al. (2023);
GPT-2 weights are downloaded from HuggingFace at run time and are not included.
