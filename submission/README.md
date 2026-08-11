# Submission package — *Neural Networks* (Elsevier)

Everything needed to submit at <https://submit.elsevier.com/NEUNET>.
Rebuild with:

```bash
bash scripts/build_submission.sh
```

which checks the journal's formal requirements, compiles the manuscript and the
cover letter, and assembles a self-contained `upload/` directory.

## What to upload, and as what

| File | Editorial Manager item type | Notes |
|---|---|---|
| `upload/manuscript.pdf` (and `upload/source/`) | Manuscript | The PDF is for review; the LaTeX source in `upload/source/` is the editable file the journal requires for typesetting. |
| `upload/Figure_1.pdf` … `Figure_4.pdf` | Figure | Vector PDF, numbered in citation order, as the guide asks. |
| `upload/cover_letter.pdf` | Cover Letter | Also proposes the journal Section (see below). |
| `upload/highlights.txt` → convert to `.docx` | Highlights | 3–5 bullets, ≤ 85 characters each. The filename must contain the word "highlights" and the file must be editable (`.docx`). |
| Declaration of competing interest | Declaration of Interest | **Generate with Elsevier's tool** at <https://declarations.elsevier.com/>, select "I have nothing to declare", download the `.docx` and upload it. Signatures are not required. |
| `upload/abstract.txt` | *(not uploaded)* | Paste-in copy of the abstract — one line of plain text, 250 words — and the six keywords, for the web forms. Generated from the manuscript by `scripts/make_abstract_txt.py`, so it cannot drift. |

Wording for every declaration is in [`declarations.md`](declarations.md); the same
statements are also printed in the manuscript.

## Before you press submit

1. **Choose the journal Section.** *Neural Networks* requires one of: Cognitive
   Science; Neuroscience; Learning Systems; Mathematical and Computational
   Analysis; Engineering and Applications. The cover letter proposes
   **Mathematical and Computational Analysis**, with **Learning Systems** as the
   alternative.
2. **Paste the Zenodo DOI into the *Research data* field.** The journal applies
   Option C of Elsevier's research data policy: the data must be deposited and
   cited. The deposit is live at <https://doi.org/10.5281/zenodo.21891261>
   (`10.5281/zenodo.21891261`) and the manuscript already cites it.
3. **Run the declarations tool** and attach the generated `.docx`.
4. **Convert `highlights.txt` to `.docx`** (the journal requires an editable
   file); keep "highlights" in the filename.
5. **Check the corresponding-author details** — the submission checklist asks for
   a full postal address and a phone number, which are not in the manuscript.

## Requirements this package already satisfies

Verified automatically by `scripts/check_submission.py`:

- Abstract **250 words** (limit 250), unstructured, no citations.
- **6 keywords** (1–6 allowed).
- Highlights: **5 bullets**, longest **77 characters** (limit 85).
- References: **author–year (APA 7th)** via `elsarticle`'s `authoryear` option and
  `elsarticle-harv.bst`, as the journal requires — *not* the numbered style.
- Every figure is cited in the text and supplied as a separate vector file.
- Author names and affiliation are included: the journal uses **single
  anonymized** review, so the manuscript is *not* anonymized.
- Declarations of competing interest, funding, CRediT roles, generative-AI use
  and data availability all present.

The journal sets no length limit for a regular Article, requires neither line
numbers nor double spacing (line numbers are included anyway, as a courtesy to
reviewers), and does not use graphical abstracts.

## Relationship to the rest of the repository

`manuscript.tex` is a thin `elsarticle` wrapper. The abstract, the body, the
appendix and the auto-generated numbers and tables are shared verbatim with the
preprint in [`../paper`](../paper), so the two versions cannot drift apart. Every
number in the manuscript is generated from the experiment outputs by
`experiments/make_figures.py`; none is typed by hand.

`journal_requirements_raw.json` is the raw record of the Guide-for-Authors check
that produced the checklist above, kept so the claims here can be audited.
