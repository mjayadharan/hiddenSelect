# 01 — dataset provenance + loss-function sanity at p* vs random p.
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl"))
using DifferentialEquations, LinearAlgebra, Statistics, Random, Logging, ForwardDiff

C = load_pipeline!()
rows = NamedTuple[]
up = []
for ws in (1, 5, 25, 100)
    Jt = forward_simulation_loss_windows([C.data[1],C.data[2]], up, C.fhn_p, odefun, C.data,
            0.0, C.δt, C.S, 0.0, ws, false; silent_=true)
    Random.seed!(4242)   # 8 random draws; the report quotes the RANGE over them
    Jr = [forward_simulation_loss_windows([C.data[1],C.data[2]], up, 0.1*randn(20), odefun,
            C.data, 0.0, C.δt, C.S, 0.0, ws, false; silent_=true) for _ in 1:8]
    push!(rows, (window_size=ws, J_true=Jt, J_random_min=minimum(Jr), J_random_max=maximum(Jr),
                 n_random_draws=8))
end
write_csv(joinpath(RESULTS, "loss_sanity.csv"), rows)

# AD path through the windowed loss
g = ForwardDiff.gradient(x -> forward_simulation_loss_windows(view(x,1:2), up, view(x,3:length(x)),
        odefun, C.data, 0.0, C.δt, C.S, C.γ2, 5, false; silent_=true), [C.data[1:2]; C.fhn_p])

write_json(joinpath(RESULTS, "data_summary.json"), Dict(
        "n_data_points" => C.Nd, "Delta_t" => C.Δt, "delta_t" => C.δt, "S" => C.S,
        "gamma2" => C.γ2, "noise_sigma_rel" => 0.05, "data_seed" => 1287436679,
        "v_range" => collect(extrema(C.data[1:2:end])),
        "w_range" => collect(extrema(C.data[2:2:end])),
        "grad_norm_at_ptrue_ws5" => norm(g), "grad_all_finite" => all(isfinite, g),
        "frozen_mod7_sha256" => depssha(MOD7)))
println("01 done")
