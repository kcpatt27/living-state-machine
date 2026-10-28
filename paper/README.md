# LSM LaTeX manuscript

- `main.tex` — arXiv-style article source
- `figure1.png` — control-loop figure
- `main.pdf` — compiled review copy (Tectonic 0.17.0, 7 pages)

## Compile

```bash
tectonic main.tex
# or:
pdflatex main.tex
pdflatex main.tex
```

The source uses `thebibliography`, so no BibTeX pass is required. Keep `figure1.png` beside `main.tex`.

## Before arXiv

- Reconcile every table with the run IDs and raw artifacts. Done 2026-10-27:
  Tables 1-2 map to `experiments/`, Table 3 to `docs/phase1_report.md` (see
  `REPRODUCE.md`). Re-check if any run is added.
- Expand the related-work section; the current seven references are a starting point.
- Add the public repository/artifact link once the LSM repo is public.
- License decided: CC BY 4.0 for the paper (see `LICENSE`), MIT for the code.
  Category: `cs.LG`, with `cs.CL`/`cs.AI` only if appropriate.
- Complete the AI-assistance statement and author review.
- Get an endorsement before the first submission.

The compiled PDF is a review copy, not a submitted arXiv record.
