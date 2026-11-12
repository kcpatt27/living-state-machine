# 01 - Fork or copy LSM for a new product

Status: open
Source: 2026-11-12 home repo proposal.
Repo: `projects/living-state-machine`.

## Evidence

LSM is a research implementation. The idea is worth pushing further as an
actual product, but the current repo carries research scaffolding, runs, and
experiment data. Keeping the product inside it would mix the reference
implementation with the thing that ships.

## Action

1. Decide fork or clean copy. A fork keeps history and the research commits; a
   clean copy starts with the smallest working slice and cites LSM as the
   reference.
2. Name the product and create its repo with a README that states the problem,
   the first user, and the first vertical slice.
3. Keep LSM as the reference implementation. Do not delete it.
4. In the `home-atlas` pass, move LSM to a `research/` folder and link the new
   product from the family manifest.

## Acceptance

A new repo exists with a name, a README, and one runnable vertical slice. LSM
remains available as the reference, and the family manifest names both.
