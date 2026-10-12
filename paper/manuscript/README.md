# Manuscript

*A Compounding Fingerprint: Telling Abstraction from Exemplar Reuse in Context*

| file | what it is |
|---|---|
| `paper.pdf` | The paper, compiled from `arxiv_upload.zip` with pdfLaTeX (MiKTeX 25.12, pdfTeX 1.40.28). |
| `main.tex` + `fig/*.pdf` | pdfLaTeX source for arXiv or Overleaf. Generated; do not edit by hand. |
| `arxiv_upload.zip` | `main.tex` plus the figure PDFs, ready to upload. |
| `sections/*.md` | **The source of the paper.** Edit these, then rebuild. Their figure links resolve from this folder (`fig/`), so GitHub shows them as broken images; read `paper.pdf` or `paper.html` instead. |
| `refs.json` | Bibliography. Every entry was checked against its arXiv, venue or GitHub page on 2026-10-11. |
| `SPEC.md` | Syntax, labels, terminology and the source-of-truth files for every number. |
| `build.py` | Builds `main.tex`, `paper.html`, `paper.pdf` and `arxiv_upload.zip` from the sections. |

## Compiling the LaTeX
`build.py` compiles `arxiv_upload.zip` with pdfLaTeX (three passes, with a fixed `SOURCE_DATE_EPOCH` so the PDF is byte-reproducible) and reports any LaTeX warnings, overfull boxes and undefined references. With MiKTeX 25.12 (pdfTeX 1.40.28) on 2026-10-12 it compiled to 44 pages with none of these. MiKTeX installs the few packages it lacks on first use. If no pdflatex is found, `build.py` prints `paper.html` with headless Chrome instead. On Overleaf or arXiv, upload `arxiv_upload.zip` and choose pdfLaTeX.

## Rebuild
```
python build.py            # CPU Python, standard library only; pdflatex optional
```
The builder refuses unknown citation keys, unknown labels, raw HTML or LaTeX outside math, ragged tables and unmapped Unicode characters.

## Provenance
- Every number comes from the generated result files listed in `SPEC.md`.
- The text was drafted from the section drafts in `../*.md` (see `../README.md`).
- An independent verification pass then confirmed 53 corrections, and a final check confirmed 18 more. All are applied.
