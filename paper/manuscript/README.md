# Manuscript

*A Compounding Fingerprint: Telling Abstraction from Exemplar Reuse in Context*

| file | what it is |
|---|---|
| `paper.pdf` | Readable PDF, printed from `paper.html` by headless Chrome. |
| `main.tex` + `fig/*.pdf` | pdfLaTeX source for arXiv or Overleaf. Generated; do not edit by hand. |
| `arxiv_upload.zip` | `main.tex` plus the figure PDFs, ready to upload. |
| `sections/*.md` | **The source of the paper.** Edit these, then rebuild. Their figure links resolve from this folder (`fig/`), so GitHub shows them as broken images; read `paper.pdf` or `paper.html` instead. |
| `refs.json` | Bibliography. Every entry was checked against its arXiv, venue or GitHub page on 2026-10-11. |
| `SPEC.md` | Syntax, labels, terminology and the source-of-truth files for every number. |
| `build.py` | Builds `main.tex`, `paper.html`, `paper.pdf` and `arxiv_upload.zip` from the sections. |

## Compiling the LaTeX
On Overleaf, upload `arxiv_upload.zip` and set the compiler to pdfLaTeX. `main.tex` was checked statically: all citations and cross-references resolve, it is pure ASCII, and its braces and math delimiters balance. It has not yet been compiled with a TeX engine, so the first build may show small layout warnings.

## Rebuild
```
python build.py            # CPU Python, standard library only
```
The builder refuses unknown citation keys, unknown labels, raw HTML or LaTeX outside math, ragged tables and unmapped Unicode characters.

## Provenance
- Every number comes from the generated result files listed in `SPEC.md`.
- The text was drafted from the section drafts in `../*.md` (see `../README.md`).
- An independent verification pass then confirmed 53 corrections, and a final check confirmed 18 more. All are applied.
