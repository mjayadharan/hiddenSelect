# R003

Permanent report ID: **R003**
Folder: `2026-09-11-R003-tighter-error-bounds`
Date: 2026-09-11

Title: Sharper error bounds for multiple shooting with the same machinery — the manuscript's
original Lipschitz bounds (Lemmas 1–2, Proposition 1, Theorem 1), proposed replacements
(logarithmic norm, residual-based constant, expectation over noise, per-window integrated
exponents in a weighted norm), and how much each change buys on FitzHugh–Nagumo.

Supersedes: none. Extends: R002 (uses its Lemma 1/2 probes and measured constants as inputs).
Inherits protocol from: R002 (FHN data recipe, probes, constants) — inputs copied into
`external_data/` with hashes in `INPUTS.md`.

## Notes
- Short-form report by request (under 4 pages, no separate definitions section; notation is
  restated in one paragraph before the results).
- Python-only analysis; `deps/` empty (nothing imported). No Julia stage.
