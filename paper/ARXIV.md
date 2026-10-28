# Getting LSM onto arXiv — and repeating the process

Status: **draft only**. The current PDF is a 7-page technical report prepared from the staged research record. It is not yet an arXiv-ready manuscript.

## What arXiv actually requires

Verified against arXiv help on 2026-10-20:

- A **registered author** account. First-time submitters, or first submissions to a new category, normally need an **endorsement** from an established arXiv author.
- A **topical, refereeable scientific contribution** with scholarly form: sections, figures, tables, references, neutral tone.
- **TeX/LaTeX is strongly preferred**. PDF is accepted, but submitting LaTeX source preserves the paper as an arXiv HTML/TeX artifact and is the better long-term choice.
- A license. The choice is **irrevocable** for that version. CC BY 4.0 matches this project's public content; the arXiv perpetual non-exclusive license is the conservative alternative.
- Disclosure of significant generative-AI use. AI tools cannot be authors; the human author takes responsibility for every claim and citation.
- Moderation, not peer review. Moderators can decline or reclassify; there is no reviewer feedback and no guarantee of acceptance.

## Current LSM draft: not ready

| Requirement | Status |
| --- | --- |
| Claims reconciled with source | Partial — tables match the DT-001 phase reports; raw run artifacts still need a check |
| Public code/artifact link | No — LSM repo currently 404 |
| LaTeX source | Yes — `papers/latex/main.tex`, compiled to 6 pages |
| Related work/citation pass | Thin — 7 references; expand before submission |
| Author/source verification | Partial — tables match the DT-001 phase reports; raw run artifacts pending |
| AI-use disclosure | Yes, footnote; needs a proper statement |
| License decision | Pending |
| Endorsement | Not started |
| arXiv account/ORCID | Not confirmed |

The draft is a good **technical report**. It is not yet a manuscript that should be uploaded.

## What to fix before submitting

1. **Reconcile every number.** Compare each table against the repo's `docs/phase1_report.md`, `docs/phase2_report.md`, exit checklists, and run directories. Remove any number that cannot be traced to a run.
2. **Make the repo public** or provide an artifact package. Include a README, license, reproduction commands, configs, and a map from each table to its run.
3. **Convert to LaTeX.** Use a clean article template, vector figures, `booktabs` tables, a real bibliography, and page numbers. Remove the draft banner and portfolio language.
4. **Expand related work.** Build a citation matrix around RAG, memory-augmented agents, long-context models, state-space models, habituation, and catastrophic forgetting. Verify every reference; do not rely on generated citations.
5. **Add the AI-assistance statement.** State that AI tools assisted with drafting/analysis and that the author verified claims and takes responsibility.
6. **Internal red-team review.** Check the statistics, seed counts, baseline fairness, negative-result framing, and the v2 “stress test, not success metric” claim.
7. **Choose the category and license.** Likely primary `cs.LG`; cross-list `cs.CL` or `cs.AI` only if the moderators agree. Choose CC BY 4.0 or the arXiv license deliberately.
8. **Get an endorsement.** If you have an institutional email or a prior arXiv paper, use that path. Otherwise ask an arXiv author who knows the area — one or two people, not a mass email. A workshop submission can also create a review record and a natural endorsement path.
9. **Submit, compile, inspect, and announce.** Upload TeX, check the compiled PDF, enter metadata, then submit. The announcement is not peer review.

## Endorsement, plainly

The first paper is the hard one. arXiv requires endorsement for first-time submitters or new categories. An institutional email or a claimed prior paper can reduce friction; as an independent researcher without either, you need a personal endorser. Ask someone who knows your work or has seen the paper. Do not spam authors. If you have no academic contact, a workshop submission is often the better first move: it provides review, a program committee, and a credible reason for someone to endorse the next preprint.

## DT-001 split

Use **dt-001** for the work that needs the private environment:

- Re-run the LSM evaluation and regenerate tables from the actual repo/run directories, if the LSM source and runs live there.
- Record the eyewright demo, because the private source and `laya` services live there.
- Capture raw logs, screenshots, and terminal traces.

Do **not** make dt-001 the only copy of the paper:

- Keep the paper in git on a branch (for example `paper/lsm`) and sync it back to the laptop.
- Write/format/compile the LaTeX and handle arXiv metadata on the laptop or a synced worktree.
- The arXiv submission itself is a browser task; dt-001 is not required for it.
- The handoff notes say dt-001's SSH/Ollama ports were closed from the laptop, so either work locally on dt-001 or deliberately open the services. Do not expose keys or private data.

## Repeatable paper protocol (papers 2 and 3)

1. **Claim ledger:** question, claims, evidence, confidence, explicit non-claims.
2. **Reproduction:** run the artifact from a clean checkout; freeze tables and figures.
3. **Related-work matrix:** claim → prior work → delta; verify references.
4. **Outline before prose:** method, setup, results, negative results, limits.
5. **LaTeX template:** reuse the same section skeleton and table macros for every paper.
6. **Red-team review:** one pass for method/statistics, one for claims/citations, one for reproducibility.
7. **Artifact release:** public repo, license, README, run commands, version tag/DOI.
8. **Submission:** arXiv preprint first if appropriate, then a workshop/conference for actual peer review.
9. **Response log:** keep reviewer comments, responses, and changes in a durable file for the next paper.
10. **Post-publication:** update CV, site, ORCID, and the project record with the arXiv/DOI link.

## Bragging-rights language

Say **“arXiv preprint”** or **“technical report”**, not “peer-reviewed publication.” A preprint is citable and useful for hiring, but it is not the same signal as a reviewed workshop or journal paper. For the second and third papers, aim at a workshop or venue with real review; use arXiv as the distribution layer, not the credential.
