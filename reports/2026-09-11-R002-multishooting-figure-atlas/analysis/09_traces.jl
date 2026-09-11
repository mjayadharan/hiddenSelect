# 09 — optimiser traces (cost vs Nelder–Mead iteration) for a representative seed (seed 2:
#      seed 1 starts on the blow-up plateau and has no trace to show), GP vs control.
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))
rhs! = make_rhs(FHN_LIB); D = make_dataset(rhs!, FHN_P, [1.0, 1.0])
FULL = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100]
rows = NamedTuple[]
for mode in (:propagate, :reset)
    r = run_sweep(rhs!, D.data, FHN_P, FULL; seed=2, mode=mode, trace_every=1)
    for (κ, tr) in r.traces, (it, v) in enumerate(tr)
        push!(rows, (arm=String(mode), seed=2, window_size=κ, iteration=it-1, J=v))
    end
end
write_csv(joinpath(RESULTS, "traces_seed2.csv"), rows)
println("09 done: ", length(rows), " trace rows")
