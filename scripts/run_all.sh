#!/usr/bin/env bash
# Full pipeline.  Each stage is skipped if its output already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
R=${R:-400}

echo "=== 1. train the tiny transformers ==="
[ -f results/models/summary.json ] || $PY -u experiments/train_tiny_models.py --out results/models --steps 6000
[ -f results/models/tt_d_induction_big.pt ] || $PY -u experiments/train_tiny_models.py --out results/models --only d --steps 8000

echo "=== 2. per-instance score matrices (GPU) ==="
[ -f results/tiny/meta.json ] || $PY -u experiments/tiny_score_matrices.py --out results/tiny --N 60000 --batch 10000
[ -f results/ioi/ioi_scores.npz ] || $PY -u experiments/ioi_score_matrix.py --out results/ioi --n_per_template 200 --batch 200
[ -f results/ioi/ioi_greedy.npz ] || $PY -u experiments/ioi_greedy.py --out results/ioi --n_search 200 --n_eval 2000

echo "=== 3. statistics (CPU / small GPU) ==="
[ -f results/synthetic/synthetic.json ] || $PY -u experiments/synthetic_study.py --out results/synthetic
[ -f results/tiny/tiny_analysis.json ]  || $PY -u experiments/tiny_analysis.py  --scores results/tiny --out results/tiny --R "$R"
[ -f results/ioi/ioi_analysis.json ]    || $PY -u experiments/ioi_analysis.py   --scores results/ioi/ioi_scores.npz --out results/ioi --R "$R"
[ -f results/ioi/greedy_analysis.json ] || $PY -u experiments/greedy_analysis.py
[ -f results/adaptive/adaptive.json ]   || $PY -u experiments/adaptive_study.py --out results/adaptive --R 60

echo "=== 4. figures, tables and the auto-generated numbers ==="
$PY -u experiments/make_figures.py --results results --figdir figures --gendir paper/generated
$PY -u experiments/write_manifest.py

echo "=== 5. paper ==="
( cd paper && latexmk -pdf -interaction=nonstopmode main.tex )
echo "done -> paper/main.pdf"
