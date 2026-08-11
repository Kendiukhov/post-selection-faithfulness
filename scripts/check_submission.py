"""Check the submission against the Neural Networks author requirements.

Requirements encoded here (from the journal's Guide for Authors):
  * abstract at most 250 words, no citations;
  * 1 to 6 keywords;
  * highlights: 3 to 5 bullets, each at most 85 characters including spaces;
  * every figure cited in the text;
  * author-year (APA) bibliography style.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def read(p: str) -> str:
    with open(os.path.join(ROOT, p)) as f:
        return f.read()


def expand_macros(text: str, numbers_tex: str) -> str:
    """Substitute the auto-generated \\MyXxx macros by their values."""
    values = dict(re.findall(r"\\newcommand\{\\(My[A-Za-z]+)\}\{(.*)\}", numbers_tex))
    for _ in range(3):
        for k, v in values.items():
            text = text.replace("\\" + k + "{}", v).replace("\\" + k + " ", v + " ")
    return text


def strip_tex(text: str) -> str:
    text = re.sub(r"^\s*%.*$", "", text, flags=re.M)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\$[^$]*\$", "MATH", text)
    text = re.sub(r"\\[A-Za-z]+\*?(\{[^}]*\})?", " ", text)
    text = text.replace("\\%", "%").replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    ok = True
    numbers = read("paper/generated/numbers.tex")

    # --- abstract ---------------------------------------------------------
    abstract = strip_tex(expand_macros(read("paper/abstract.tex"), numbers))
    words = len(abstract.split())
    print(f"abstract: {words} words (limit 250)")
    if words > 250:
        print("  FAIL: abstract exceeds the 250-word limit")
        ok = False
    if re.search(r"\\cite", read("paper/abstract.tex")):
        print("  FAIL: abstract contains a citation")
        ok = False

    # --- keywords ---------------------------------------------------------
    ms = read("submission/manuscript.tex")
    kw = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", ms, flags=re.S)
    if kw:
        n_kw = len([k for k in kw.group(1).split("\\sep") if k.strip()])
        print(f"keywords: {n_kw} (allowed 1-6)")
        if not 1 <= n_kw <= 6:
            print("  FAIL: keyword count outside 1-6")
            ok = False

    # --- highlights -------------------------------------------------------
    hl_path = os.path.join(ROOT, "submission", "highlights.txt")
    if os.path.exists(hl_path):
        bullets = [
            ln.strip().lstrip("-").strip()
            for ln in open(hl_path).read().splitlines()
            if ln.strip().startswith("-")
        ]
        print(f"highlights: {len(bullets)} bullets (allowed 3-5)")
        if not 3 <= len(bullets) <= 5:
            print("  FAIL: highlight count outside 3-5")
            ok = False
        for b in bullets:
            flag = "" if len(b) <= 85 else "  <-- FAIL: over 85 characters"
            print(f"  {len(b):3d} chars: {b}{flag}")
            if len(b) > 85:
                ok = False
    else:
        print("highlights: file missing")
        ok = False

    # --- figures cited ----------------------------------------------------
    import glob

    body = read("paper/body.tex") + "".join(
        open(f).read() for f in glob.glob(os.path.join(ROOT, "paper/generated/*.tex"))
    )
    for fig in ["fig:anatomy", "fig:tiny", "fig:ioi", "fig:greedy"]:
        n_ref = len(re.findall(re.escape("\\ref{" + fig + "}"), body))
        if n_ref == 0:
            print(f"  FAIL: figure {fig} is never cited in the text")
            ok = False
    print("figures: all four cited in the text" if ok else "figures: see failures above")

    # --- bibliography style ----------------------------------------------
    if "elsarticle-harv" in ms and "authoryear" in ms:
        print("bibliography: author-year (APA), elsarticle-harv -- correct for this journal")
    else:
        print("  FAIL: journal requires author-year (APA) references")
        ok = False

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
