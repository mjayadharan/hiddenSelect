# R002 — frozen code snapshots (`deps/`)

Two byte-identical snapshots of the repository's Julia sources:

- `deps/*.jl` — **post-fix** sources: the working tree after the R002 bug fixes (parent commit `44abf4a4a14594d5e6761d71e9aa1fedd219dc28`; the fixes are committed together with this report). Every experiment (`analysis/01…09`) runs from these through `analysis/common.jl` (`depspath`) and `analysis/hs_core.jl`.
- `deps/prefix/*.jl` — **pre-fix** sources, byte-identical to commit `44abf4a4a14594d5e6761d71e9aa1fedd219dc28` (`origin/main` at the time of freezing; `git fetch` was run first — lesson from R001). Used only by `analysis/06_bugs.jl` (`prefixpath`) to quantify the defects.

`integrator.jl` and `helper_functions.jl` are identical in both snapshots (unchanged by the fixes). No analysis script reads the live tree; `analysis/verify_R002.py` gate G1 asserts these hashes.

| file | SHA-256 | bytes | lines |
|---|---|---|---|
| `fhn_model_selection_mod 6 stability analysis and visualization.jl` | `763e765818076f4f0287af1d94b307c662e505cd2537feda3781a8cb935bd58f` | 41587 | 1083 |
| `fhn_model_selection_mod 7.jl` | `7d202e84bb8c7071654770185397a309e1e5388f98488b391cdf9e00640cadd3` | 34107 | 883 |
| `fhn_model_selection_mod 8.jl` | `849105fcbeccc98fc5487bf3f17b9e78c17fb6ea1b14b1e0011afa44ac758f32` | 35727 | 865 |
| `helper_functions.jl` | `a3f4494aa5d2b6bf180e5a1aa55ed5b2cc0184deb153e1714bca8abe1b06df1d` | 1569 | 50 |
| `integrator.jl` | `368a5242f5a82fc1988cdd010edbbff1ae7e1116c24ab6d2b7de32bad6fa827e` | 10685 | 289 |
| `plotting_window_size.jl` | `4b8a7fb8048d29faa74ccdb7148c14d4025172a88cd070005e20f005ac9bd855` | 29580 | 653 |
| `prefix/fhn_model_selection_mod 6 stability analysis and visualization.jl` | `a8785cf4814f649c84e943d0ea84d03cfc990136606102c12a36695a9d7a1577` | 41312 | 1080 |
| `prefix/fhn_model_selection_mod 7.jl` | `899551101971cf3b6ab55e4d2a31a77245c2c377c92219cf0f351e15b67b65b0` | 33841 | 881 |
| `prefix/fhn_model_selection_mod 8.jl` | `545d25540d9376dd5d653bb2229e42a408a855c7db78f348bdc46230e21fd3e3` | 34594 | 847 |
| `prefix/helper_functions.jl` | `a3f4494aa5d2b6bf180e5a1aa55ed5b2cc0184deb153e1714bca8abe1b06df1d` | 1569 | 50 |
| `prefix/integrator.jl` | `368a5242f5a82fc1988cdd010edbbff1ae7e1116c24ab6d2b7de32bad6fa827e` | 10685 | 289 |
| `prefix/plotting_window_size.jl` | `bb67733a975c2fcc14d96bb6a720aaebed6157d5cd8d3864772e69a0c17ce490` | 29458 | 652 |
