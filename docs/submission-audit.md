# Submission audit: Living State Machine

Date: 2026-11-24. Audited commit: `f000cee` on `main` (topic branch
`mining/shard8-2026-11-24` is identical at the time of the audit).
Scope: the arXiv submission surface only. The research claims are out of scope;
the numbers are treated as honest per `docs/paper_abstract.md` and the run
artifacts it cites.

## Verdict

**Not ready to submit as documented. Ready as a review copy.**

The research and the numbers are in good shape. The submission surface is split:
the manuscript is on `main`, the abstract says it is on a `paper/arxiv` branch
that does not exist, and two items from the paper's own pre-submission checklist
are still open.

## Evidence

| Check | Command | Result |
|---|---|---|
| Manuscript present on `main` | `Test-Path paper/main.tex, paper/main.pdf` | both true |
| Abstract names the manuscript's branch | `docs/paper_abstract.md` line 5 | says "`paper/main.pdf` on branch `paper/arxiv`" |
| That branch exists | `git branch -a \| Select-String paper` | no match; only `main` and `origin/main` |
| Paper checklist state | `paper/README.md` "Before arXiv" | tables reconciled 2026-10-27; license decided; AI statement, repo link, endorsement open |
| AI-assistance statement in the manuscript | `Select-String paper/main.tex -Pattern 'AI\|assist\|Cursor'` | no statement found; the author block names only the author |
| Repository link in the manuscript | `Select-String paper/main.tex -Pattern 'github'` | no repo URL in the source |
| Compile recipe | `paper/README.md` | `tectonic main.tex` (8 pages, Tectonic 0.17.0) |
| Review-copy status | `paper/README.md` last line | "The compiled PDF is a review copy, not a submitted arXiv record" |

The repo is public (`github.com/kcpatt27/living-state-machine`), so the "add the
public repository link" item is unblocked.

## Remaining gates, in order

1. **Fix the branch claim.** Either recreate `paper/arxiv` at the manuscript
   commit and name it in the abstract, or change `docs/paper_abstract.md` to say
   the manuscript is on `main`. A reader who runs `git checkout paper/arxiv` gets
   a failure and stops trusting the rest of the document.
2. **Add the AI-assistance statement to `main.tex`.** The paper checklist
   requires it and the abstract advertises it ("AI pair-programming (Cursor);
   every claim is backed by committed configs"). Put it in an
   `\section*{Acknowledgments}` or a footnote, with the scope of the assistance.
3. **Add the repository URL to the manuscript** (data and code availability
   paragraph). The repo is public; no reason to wait.
4. **Re-check Tables 1-3 against the run artifacts.** The 2026-10-27
   reconciliation predates the later research commits (`20e6ce6`, `0cbf6f8`,
   `809e6fa`). Confirm no run was added or renumbered since.
5. **Endorsement for `cs.LG`.** External and not a repo gate. Track it as
   pending, not as a blocker in the repo.

## Discrepancies

- `docs/paper_abstract.md` line 5 names a branch that does not exist.
- `README.md` says "Preprint: `paper/main.pdf` - arXiv `cs.LG`, pending
  endorsement", which reads as submitted-with-endorsement-pending; the paper
  README says the PDF is a review copy. Reword one of them to match the actual
  state (not submitted).
- The paper checklist still lists the repo link as open while the abstract
  already links the repo. One of the two is stale.

## What this audit did not do

- It did not re-run the test suite or the evals. The numbers are taken from the
  committed docs; re-running them is a separate gate on the slowest supported
  host.
- It did not rebuild `main.pdf`. The review copy predates the two docs edits
  above, so rebuild after the fixes.
- It did not submit anything and did not contact arXiv.

## After the fixes

Re-run the evidence table above, rebuild the PDF, and record the new commit sha
at the top of this file. The verdict moves to "ready" only when items 1 to 4 are
done; item 5 is external.

## Related

- `paper/README.md` for the pre-submission checklist.
- `docs/paper_abstract.md` for the claims the submission rests on.
- `notes-vault/ideas/2026-11-24-release-readiness-report.md` for the report
  shape (verdict, evidence, gates, commands, discrepancies).
