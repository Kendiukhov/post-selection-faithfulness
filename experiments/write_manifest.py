"""Record everything needed to reproduce the reported numbers exactly."""

from __future__ import annotations

import hashlib
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
import platform
import subprocess
import sys

FILES = [
    "results/models/summary.json",
    "results/tiny/meta.json",
    "results/tiny/tiny_analysis.json",
    "results/ioi/meta.json",
    "results/ioi/ioi_analysis.json",
    "results/ioi/greedy_meta.json",
    "results/ioi/greedy_analysis.json",
    "results/synthetic/synthetic.json",
    "results/adaptive/adaptive.json",
    "paper/generated/numbers.tex",
]

BIG = [
    "results/tiny/tt_a_scores.npz",
    "results/tiny/tt_a_iit_scores.npz",
    "results/tiny/tt_b_scores.npz",
    "results/tiny/tt_c_scores.npz",
    "results/ioi/ioi_scores.npz",
    "results/ioi/ioi_greedy.npz",
]


def sha(path: str, limit: int = 1 << 30) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            limit -= len(b)
            if limit <= 0:
                break
    return h.hexdigest()[:16]


def main() -> None:
    man: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for mod in ["numpy", "scipy", "torch", "transformers", "matplotlib"]:
        try:
            man[mod] = __import__(mod).__version__
        except Exception:
            man[mod] = None
    try:
        import torch

        man["accelerator"] = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    except Exception:
        man["accelerator"] = "cpu"
    try:
        man["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        man["git_commit"] = None

    man["artifacts"] = {}
    for p in FILES + BIG:
        if os.path.exists(p):
            man["artifacts"][p] = {"bytes": os.path.getsize(p), "sha256_16": sha(p)}
    with open("reproducibility_manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    print(json.dumps({k: v for k, v in man.items() if k != "artifacts"}, indent=2))
    print(f"{len(man['artifacts'])} artifacts recorded")


if __name__ == "__main__":
    main()
