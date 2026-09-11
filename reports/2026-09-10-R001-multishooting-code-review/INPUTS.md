# R001 — inputs

All inputs are inside this folder. Nothing is read from another report, and
nothing is read from the live repository tree at run time.

## Copied into `deps/` (frozen source snapshot)

There is no external *data* input. The report's input **is** the repository's
own Julia source, frozen byte-identically from the **working tree** at commit
`8648cde77cdc559ac941a5ecb518241446b758ea` (branch `main`). Seven of the eight
files match that commit exactly; `fhn_model_selection_mod 8.jl` carries the one
uncommitted whitespace-only edit present at review time, which changes no line
number any finding depends on. See `deps/MANIFEST.md` for the exact statement
and the committed-form hash.

| file | SHA-256 (first 16) | lines | used for |
|---|---|---|---|
| `integrator.jl` | `368a5242f5a82fc1` | 289 | Tsit5 tableau, cache, fixed-step `integrate` |
| `helper_functions.jl` | `a3f4494aa5d2b6bf` | 50 | multi-index / monomial-name generation |
| `fhn_model_selection_mod 2.jl` | `45136459c8a86f8d` | 515 | finding F7(i), F7(ii) — read only |
| `fhn_model_selection_mod 5.jl` | `39ac397c34185ea0` | 900 | finding F2, F4 — read only |
| `fhn_model_selection_mod 6 stability analysis and visualization.jl` | `a8785cf4814f649c` | 1080 | finding F1 (`odefun_new`, lines 980–993) |
| `fhn_model_selection_mod 7.jl` | `443b209fa28e1133` | 881 | RHS (19–23), dataset (60–107), losses (123–418) |
| `fhn_model_selection_mod 8.jl` | `38ed9a81308d1190` | 754 | `odefun_poly!` (47–77), `forward_simulation_loss` (257–339) |
| `plotting_window_size.jl` | `a3ee0db0609af199` | 535 | finding F5 — read only |

Full hashes: `deps/MANIFEST.md`. Gate `G1_deps_frozen_matches_manifest`
re-verifies every one of them on each run.

## Manifested only (large; stays at its committed location)

None. The largest artefacts the repository holds (`figs/**`, ~40 MB of PNG and
MP4) are *discussed* by this report but never read by it — the findings about
them (F1, F2, F7(i)) are derived from the producing code, not from the images.

## Dataset provenance

The FitzHugh–Nagumo dataset is **not transcribed** into this folder. It is
produced at run time by executing frozen `mod 7` lines 60–107 verbatim (line 108
begins that file's plotting block). Derived constants are recorded in
`analysis/results/data_summary.json`: $N=101$ data times, $\Delta t = 1.0$,
$\delta t = 0.1$, $S = 10$, relative noise $0.05$, RNG seed `1287436679`.

## Hash ledger

`analysis/results/results_hashes.json` records SHA-256 for every result file,
written by `analysis/verify_R001.py` (gate `G14_completeness_hashes`).
