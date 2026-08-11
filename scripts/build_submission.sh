#!/usr/bin/env bash
# Build the Neural Networks submission and assemble a self-contained upload
# package.  Run scripts/run_all.sh first so that paper/generated is current.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}

echo "=== checking the journal's formal requirements ==="
$PY scripts/check_submission.py

echo "=== building the manuscript and the cover letter ==="
( cd submission && latexmk -pdf -interaction=nonstopmode manuscript.tex >/dev/null )
( cd submission && latexmk -pdf -interaction=nonstopmode cover_letter.tex >/dev/null )

echo "=== assembling submission/upload/ ==="
OUT=submission/upload
rm -rf "$OUT" && mkdir -p "$OUT"

# Elsevier wants each figure as its own file, named in citation order.
i=1
for f in fig_anatomy fig_tiny fig_ioi fig_greedy; do
  cp "figures/$f.pdf" "$OUT/Figure_$i.pdf"
  i=$((i+1))
done

# A self-contained LaTeX source tree: the class file lives on Elsevier's side,
# everything else travels with the manuscript.
mkdir -p "$OUT/source/generated"
cp submission/manuscript.tex "$OUT/source/"
cp paper/macros.tex paper/abstract.tex paper/body.tex paper/appendix.tex \
   paper/protocol.tex paper/limitations.tex paper/references.bib "$OUT/source/"
cp paper/generated/*.tex "$OUT/source/generated/"
cp figures/*.pdf "$OUT/source/"
# flatten the \paperdir / \gendir indirection for the standalone copy
$PY - "$OUT/source/manuscript.tex" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace("{../paper/generated/}", "{generated/}").replace("{../paper/}", "{}")
s = s.replace("\\graphicspath{{../figures/}}", "\\graphicspath{{./}}")
open(p, "w").write(s)
PYEOF

cp submission/manuscript.pdf "$OUT/manuscript.pdf"
cp submission/cover_letter.pdf "$OUT/cover_letter.pdf"
cp submission/highlights.txt "$OUT/highlights.txt"
cp submission/declarations.md "$OUT/declarations.md"

echo "=== verifying the standalone source compiles ==="
( cd "$OUT/source" && latexmk -pdf -interaction=nonstopmode manuscript.tex >/dev/null \
  && echo "standalone build OK: $(pdfinfo manuscript.pdf | awk '/Pages/{print $2}') pages" )
( cd "$OUT/source" && latexmk -c >/dev/null 2>&1 || true )

echo
echo "upload package ready in $OUT:"
ls -1 "$OUT"
