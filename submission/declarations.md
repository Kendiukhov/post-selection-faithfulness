# Author declarations

Text for the statements *Neural Networks* requires. All of these also appear in
the manuscript itself (`manuscript.tex`, after the Conclusion); this file exists
so that the wording can be pasted into the submission system and into the
separate files the journal asks for.

---

## Declaration of competing interest

> The author declares that he has no known competing financial interests or
> personal relationships that could have appeared to influence the work reported
> in this paper.

**How to submit it.** *Neural Networks* requires this to be produced with
Elsevier's declarations tool at <https://declarations.elsevier.com/>, downloaded
as a `.doc`/`.docx`, and uploaded as a separate file at the *Attach/upload files*
step. Select **"I have nothing to declare"**. Author signatures are not required.

---

## Funding

> This research received no specific grant from any funding agency in the public,
> commercial, or not-for-profit sectors.

---

## CRediT authorship contribution statement

> **Ihor Kendiukhov:** Conceptualization, Methodology, Software, Validation,
> Formal analysis, Investigation, Data curation, Writing – original draft,
> Writing – review and editing, Visualization, Project administration.

---

## Declaration of generative AI and AI-assisted technologies in the writing process

> During the preparation of this work the author used a large language model
> (Anthropic Claude) to assist with implementing the software, running the
> experiments, and drafting and editing the text. After using this tool the
> author reviewed and edited the content as needed and takes full responsibility
> for the content of the publication.

This statement follows Elsevier's required template and is placed in a declared
section at the end of the manuscript, before the references, as the policy
specifies. It does not apply to the use of AI tools as an object of study.

---

## Data availability

*Neural Networks* applies **Option C** of Elsevier's research-data policy: data
must be deposited in a repository and cited in the article, or the reason for not
sharing must be stated.

> All code, the cached per-instance faithfulness score matrices, the analysis
> outputs and the manuscript source are publicly available at
> <https://github.com/Kendiukhov/post-selection-faithfulness> and are archived at
> Zenodo (DOI: `10.5281/zenodo.XXXXXXX`). No third-party data were used: the only
> external artefact is the publicly available GPT-2 small checkpoint, downloaded
> at run time from the Hugging Face Hub; all evaluation prompts and all small
> transformers are generated or trained by the released code.

**Before submitting:** create a Zenodo deposit from the GitHub repository (Zenodo
has a one-click GitHub integration), then replace `10.5281/zenodo.XXXXXXX` above,
in `manuscript.tex`, and in the `[dataset]` reference below with the minted DOI,
and paste the DOI into the *Research data* field of the submission form.

Dataset reference to add to the bibliography once the DOI exists:

```bibtex
@misc{kendiukhov2026psfdata,
  author       = {Kendiukhov, Ihor},
  title        = {Post-selection-valid confidence bounds for mechanistic
                  faithfulness: code and per-instance score matrices},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  note         = {[dataset]}
}
```

---

## Originality and prior publication

> The manuscript is original, is not under consideration by any other journal,
> and has not previously been published in any form, including as a conference
> paper.
