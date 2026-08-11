#!/usr/bin/env python3
"""Write submission/abstract.txt: the abstract as one line of plain text,
followed by the keywords.

Editorial Manager asks for the abstract and the keywords in web forms, not as
files, so they have to be pasted.  Deriving them here from paper/abstract.tex
and submission/manuscript.tex means the pasted text cannot drift from the
manuscript: the auto-generated numbers are resolved from paper/generated, and
nothing is retyped.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plain_abstract() -> str:
    numbers = open(os.path.join(ROOT, "paper", "generated", "numbers.tex")).read()
    values = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", numbers))

    src = open(os.path.join(ROOT, "paper", "abstract.tex")).read()
    text = "\n".join(l for l in src.splitlines() if not l.startswith("%"))

    text = re.sub(r"\\(My\w+)(?:\{\})?", lambda m: values[m.group(1)], text)
    text = re.sub(r"\\emph\{(.*?)\}", r"\1", text, flags=re.S)
    text = text.replace("$95\\%$", "95%").replace("$n=1000$", "n=1000")
    text = text.replace("\\%", "%").replace("\\,", " ")
    text = re.sub(r"\s+", " ", text).strip()

    leftover = re.findall(r"\\\w+|\$", text)
    if leftover:
        sys.exit(f"unhandled LaTeX in the abstract: {sorted(set(leftover))}")
    return text


def keywords() -> list:
    src = open(os.path.join(ROOT, "submission", "manuscript.tex")).read()
    block = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", src, re.S).group(1)
    return [re.sub(r"\s+", " ", k).strip() for k in block.split("\\sep")]


def main() -> None:
    abstract, kw = plain_abstract(), keywords()
    out = os.path.join(ROOT, "submission", "abstract.txt")
    with open(out, "w") as f:
        f.write(abstract + "\n\n")
        f.write("Keywords: " + "; ".join(kw) + "\n")
    print(f"wrote {os.path.relpath(out, ROOT)}: "
          f"{len(abstract.split())} words, {len(kw)} keywords")


if __name__ == "__main__":
    main()
