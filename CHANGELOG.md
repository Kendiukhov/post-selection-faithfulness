# Changelog

## 0.1.0

First public release: the library, the experiments and the paper.

Notable implementation decisions that a reader may want to know about:

* The simultaneous band uses the **empirical bootstrap-t**, recomputing the
  standard deviation inside every replicate. The textbook multiplier bootstrap,
  which holds the standard deviation fixed, under-covers for binary faithfulness
  scores because the sample mean and sample standard deviation are dependent.
  See `tests/test_bounds.py::test_fixed_sigma_bootstrap_undercovers_for_binary_scores`.
* The cluster-robust band recomputes the cluster-robust standard error inside
  every wild-bootstrap replicate, as Cameron, Gelbach and Miller (2008)
  prescribe.
* Bootstrap mat-muls run on an accelerator when one is available. Set
  `PSF_BOOTSTRAP_BACKEND=numpy` to force the CPU path.
