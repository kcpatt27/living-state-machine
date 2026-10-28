# LSM LaTeX manuscript

- `main.tex` — arXiv-style article source
- `figure1.png` — control-loop figure
- `main.pdf` — compiled review copy (Tectonic 0.17.0, 6 pages)

## Compile

```bash
tectonic main.tex
# or:
pdflatex main.tex
pdflatex main.tex
```

The source uses `thebibliography`, so no BibTeX pass is required. Keep `figure1.png` beside `main.tex`.

## Before arXiv

- Reconcile every table with the run IDs and raw artifacts.
- Expand the related-work section; the current seven references are a starting point.
- Add the public repository/artifact link once the LSM repo is public.
- Decide the license (CC BY 4.0 matches the site) and category (`cs.LG`, with `cs.CL`/`cs.AI` only if appropriate).
- Complete the AI-assistance statement and author review.
- Get an endorsement before the first submission.

The compiled PDF is a review copy, not a submitted arXiv record.
