# 03 — finding F2: the stiff/DiffEq branch drops the w-residual.
# self-contained: frozen deps/ shadows the live repository tree
#
# The repaired copy is produced by a MECHANICAL edit of the frozen source: the
# orphaned continuation line `+ dot(D_2...)` is re-joined to the statement above
# it by moving the leading `+` to the end of the previous line. Nothing else is
# changed, and the function is renamed so both versions coexist.
include(joinpath(@__DIR__, "common.jl"))
using DifferentialEquations, LinearAlgebra, Statistics, Random, Logging

C = load_pipeline!()

src = depslines(MOD7)[MOD7_FSLW[1]:MOD7_FSLW[2]]
patched = replace(join(src, "\n"),
                  "function forward_simulation_loss_windows(" => "function fslw_repaired(")
lines = split(patched, "\n")
n_joins = 0
for i in eachindex(lines)
    if startswith(strip(lines[i]), "+ dot(D_2[")
        lines[i-1] = rstrip(lines[i-1]) * " +"
        lines[i]   = replace(lines[i], r"^(\s*)\+ " => s"\1  ")
        global n_joins += 1
    end
end
n_joins == 1 || error("expected exactly 1 orphaned continuation, found $n_joins")
include_string(Main, join(lines, "\n"), "deps/$MOD7:repaired")

rows = NamedTuple[]
up = []
for ws in (1, 10, 25, 100)
    asis = forward_simulation_loss_windows([C.data[1],C.data[2]], up, C.fhn_p, odefun, C.data,
             0.0, C.δt, C.S, 0.0, ws, true; solver_=Tsit5(), adaptive_=true, silent_=true)
    fixed = fslw_repaired([C.data[1],C.data[2]], up, C.fhn_p, odefun, C.data,
             0.0, C.δt, C.S, 0.0, ws, true; solver_=Tsit5(), adaptive_=true, silent_=true)
    expl = forward_simulation_loss_windows([C.data[1],C.data[2]], up, C.fhn_p, odefun, C.data,
             0.0, C.δt, C.S, 0.0, ws, false; silent_=true)
    push!(rows, (window_size=ws, stiff_as_written=asis, stiff_repaired=fixed,
                 explicit_branch=expl, missing_fraction=1 - asis/fixed,
                 repaired_minus_explicit=abs(fixed - expl)))
end
write_csv(joinpath(RESULTS, "stiff_branch.csv"), rows)
write_json(joinpath(RESULTS, "stiff_branch_meta.json"), Dict("n_continuation_joins" => n_joins,
                        "frozen_mod7_sha256" => depssha(MOD7)))
println("03 done")
