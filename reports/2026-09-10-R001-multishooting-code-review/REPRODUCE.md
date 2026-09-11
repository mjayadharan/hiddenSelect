# R001 — reproduce

## Environment

```
julia    1.11.3            (/Applications/Julia-1.11.app/Contents/Resources/julia/bin/julia)
project  hiddenSelect/Project.toml + Manifest.toml as committed
         DifferentialEquations 7.15.0, Optim 1.11.0, ForwardDiff 0.10.38,
         DataFrames 1.7.0, CairoMakie 0.13.1 (NOT used by this report)
python   3.11.7            (/opt/anaconda3/bin/python3)  — gate harness only
         pandas 2.3.3
pdflatex (TeX Live 2024), run twice from this folder
```

Machine notes:

- The analysis deliberately loads **no plotting stack**. `CairoMakie` cannot be
  precompiled under the sandbox on this machine: it rebuilds the `REPL` stdlib,
  whose precompile-statement generator opens a pseudo-terminal, and
  `/dev/ptmx` is denied (`Failed to open ptm`). Outside the sandbox this is a
  non-issue; inside it, keep the analysis Makie-free.
- If `~/.julia` is read-only (sandboxed runs), prepend a writable depot:
  `export JULIA_DEPOT_PATH="$TMPDIR/jdepot:$HOME/.julia"`.
- Julia needs no extra packages: `analysis/common.jl` carries a minimal JSON
  writer rather than adding `JSON.jl`, so the report never modifies the
  environment it is auditing.

## Ordered regeneration

```bash
cd reports/2026-09-10-R001-multishooting-code-review

# data + losses, RHS/bug checks, stiff-branch comparison, headline sweep
julia --project=../.. --startup-file=no analysis/01_data_and_loss.jl
julia --project=../.. --startup-file=no analysis/02_rhs_and_bugs.jl
julia --project=../.. --startup-file=no analysis/03_stiff_branch.jl
julia --project=../.. --startup-file=no analysis/04_sweep.jl      # the long one
# (or: julia --project=../.. --startup-file=no analysis/run_all.jl)

python analysis/verify_R001.py            # 14 gates -> results/gates_summary.json

pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Wall clock on the reference machine (single-threaded, warm precompile cache):
01 ≈ 40 s, 02 ≈ 25 s, 03 ≈ 60 s, **04 ≈ 6 min** (14 × 2500 Nelder–Mead
iterations), gates < 1 s, `pdflatex` ≈ 4 s per pass.

## Key parameters

Frozen in the driver and restated in report §2: $S=10$, $\delta t=\Delta t/S=0.1$,
$\Delta t=1.0$, $\gamma_2=5\times10^{-2}$, $N=101$, relative noise $0.05$,
$N_p=20$. Sweep: $\kappa \in \{1,2,5,10,25,50,100\}$, `NelderMead`, 2500 iterations.

## Determinism

Fully deterministic. Two RNG streams, both seeded in code:

- `Random.seed!(1287436679)` inside frozen `mod 7` line 101 — the observation
  noise. Reproduced exactly because the data block is executed from `deps/`.
- `Random.seed!(1)` in `analysis/04_sweep.jl` — the optimiser's starting point
  ($\|p_0-p^\star\|=1.5679`), and `Random.seed!(4242)` / `Random.seed!(11)` for the
  random-$p$ draws in 01 and the RHS comparison in 02.

Gate `G8_section5_table_reproduces` asserts all 14 cells of the report's
headline table against the regenerated `results/sweep.csv` to $5\times10^{-5}$;
gate `G1_deps_frozen_matches_manifest` asserts the frozen inputs have not drifted.
