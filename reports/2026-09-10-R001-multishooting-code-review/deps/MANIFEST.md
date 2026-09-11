# R001 — frozen code snapshot (`deps/`)

Byte-identical copies of the repository Julia sources this report analyses,
taken from the **working tree** at commit
`8648cde77cdc559ac941a5ecb518241446b758ea` (branch `main`).

Seven of the eight files are byte-identical to that commit. The exception is
`fhn_model_selection_mod 8.jl`, which carries the single uncommitted,
whitespace-only modification present in the working tree at review time (one
blank line inserted at line 264, inside the `if stiff_solver` branch of
`forward_simulation_loss`). Its committed form hashes to
`6060ae049fb611d74929349042c7e2b0d69c8efe3e64113a0a0fc7ad9eae67e4`. No finding
in this report depends on that line: F3 concerns lines 97 and 260, and the
whitespace changes no line number at or before 263.

Every analysis script in `analysis/` resolves `deps/` **first** (constant
`DEPS` in `analysis/common.jl`) and never reads the live tree, so the report
regenerates identically even after the live sources are edited — which matters
here because the scripts extract source **by line range**, and an edit to a
live file would otherwise shift those ranges silently.

| file | SHA-256 | bytes | lines |
|---|---|---|---|
| `fhn_model_selection_mod 2.jl` | `45136459c8a86f8d3fb40525d470055fc43b16adf5e0df2e8aac397da2886328` | 19715 | 515 |
| `fhn_model_selection_mod 5.jl` | `39ac397c34185ea01c4d1b212a71c229ac1a68084b3c65aa264a04ac5acdd171` | 35068 | 900 |
| `fhn_model_selection_mod 6 stability analysis and visualization.jl` | `a8785cf4814f649c84e943d0ea84d03cfc990136606102c12a36695a9d7a1577` | 41312 | 1080 |
| `fhn_model_selection_mod 7.jl` | `443b209fa28e113330f4221cd36a0372a6c2d07ab6c71748ad87b5586f73760a` | 33876 | 881 |
| `fhn_model_selection_mod 8.jl` | `38ed9a81308d11903ff9a4e79f0b228b6f7a6a5c8fa328a93c2a45ef6068c22b` | 30182 | 754 |
| `helper_functions.jl` | `a3f4494aa5d2b6bf180e5a1aa55ed5b2cc0184deb153e1714bca8abe1b06df1d` | 1569 | 50 |
| `integrator.jl` | `368a5242f5a82fc1988cdd010edbbff1ae7e1116c24ab6d2b7de32bad6fa827e` | 10685 | 289 |
| `plotting_window_size.jl` | `a3ee0db0609af199db4e90a362f02717b98f5c915513cb229c4f1cfad95730ee` | 23540 | 535 |
