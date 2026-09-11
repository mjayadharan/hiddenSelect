# 00 — core identity + timing: ms_loss vs the frozen mod 8 loss; benchmark.
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))
using DifferentialEquations, Logging

# Load frozen mod 8 (post-fix) RHS, params, data block and losses verbatim.
run_block(depspath(MOD8), MOD8_ODEFUN...)
run_block(depspath(MOD8), MOD8_ODEFUN_POLY...)
run_block(depspath(MOD8), MOD8_PARAMS...)
run_block(depspath(MOD8), MOD8_DATA...)
run_block(depspath(MOD8), MOD8_SMOOTHL1...)
run_block(depspath(MOD8), MOD8_FSL...)
run_block(depspath(MOD8), MOD8_FSLW...)
odefun_generic = (du,u,p,t) -> odefun_poly!(du,u,p,t; indices=multi_index_set, deg=3)

rhs! = make_rhs(FHN_LIB)
D = make_dataset(rhs!, FHN_P, [1.0, 1.0])
S = 10; δt = Δt/S; γ = 5e-2

# (1) dataset identity with the frozen mod 8 data block
data_maxdiff = maximum(abs.(D.data .- data))
println("dataset max |diff| vs frozen mod 8 block = ", data_maxdiff)

# (2) loss identity at random p and several κ (flat penalty)
rng = MersenneTwister(7); maxd = 0.0; rows = NamedTuple[]
for κ in (1, 2, 5, 10, 25, 50, 100), trial in 1:6
    p = trial == 1 ? copy(FHN_P) : FHN_P .+ 0.1 .* randn(rng, 20)
    x0 = data[1, 2:3]
    a = ms_loss(x0, p, rhs!, D.data, δt, S, γ, κ)
    b = forward_simulation_loss_windows(x0, [], p, odefun_generic, data, 0.0, δt, S, γ, κ, false; silent_=true)
    global maxd = max(maxd, abs(a-b)/max(1.0, abs(b)))
    push!(rows, (window_size=κ, trial=trial, J_core=a, J_frozen_mod8=b, absdiff=abs(a-b)))
end
write_csv(joinpath(RESULTS, "core_identity.csv"), rows)
println("max rel |ms_loss - frozen mod8 loss| = ", maxd)

# (3) RHS identity: make_rhs vs frozen odefun_poly! vs frozen old-order odefun with permuted p
perm = old_to_new_perm()
rng = MersenneTwister(11); rmax1 = 0.0; rmax2 = 0.0
for _ in 1:200
    u = 2 .* randn(rng, 2); p = randn(rng, 20)
    d1 = rhs!(zeros(2), u, p, 0.0); d2 = odefun_poly!(zeros(2), u, p, 0.0; indices=multi_index_set, deg=3)
    global rmax1 = max(rmax1, maximum(abs.(d1 .- d2)))
    # old ordering: p_old such that p_new = p_old[perm] blockwise
    p_old = similar(p); p_old[perm] .= p[1:10]; p_old[10 .+ perm] .= p[11:20]
    d3 = odefun(zeros(2), u, p_old, 0.0)
    global rmax2 = max(rmax2, maximum(abs.(d1 .- d3)))
end
println("RHS identity: vs odefun_poly! = $rmax1, vs old-order odefun = $rmax2")

# (4) timing
f = make_objective(rhs!, D.data, δt, S, γ, 5)
z = [D.data[1,2:3]; FHN_P]
f(z); t = @elapsed (for _ in 1:200; f(z); end)
fg = make_objective(rhs!, D.data, δt, S, γ, 100)
fg(z); tg = @elapsed (for _ in 1:200; fg(z); end)
cfg = ForwardDiff.GradientConfig(f, z, ForwardDiff.Chunk{12}())
g = ForwardDiff.gradient(f, z, cfg); tgrad = @elapsed (for _ in 1:20; ForwardDiff.gradient(f, z, cfg); end)
fo = x -> forward_simulation_loss_windows(view(x,1:2), [], view(x,3:22), odefun_generic, data, 0.0, δt, S, γ, 5, false; silent_=true)
fo(z); to = @elapsed (for _ in 1:50; fo(z); end)
println("loss eval: core κ=5 $(t/200*1e3) ms, κ=100 $(tg/200*1e3) ms; frozen mod8 $(to/50*1e3) ms; grad $(tgrad/20*1e3) ms")

write_json(joinpath(RESULTS, "core_check.json"), Dict(
    "dataset_max_absdiff" => data_maxdiff, "loss_max_reldiff" => maxd,
    "rhs_vs_odefun_poly_max" => rmax1, "rhs_vs_old_order_odefun_max" => rmax2,
    "ms_per_loss_eval_core_k5" => t/200*1e3, "ms_per_loss_eval_core_k100" => tg/200*1e3,
    "ms_per_loss_eval_frozen_mod8_k5" => to/50*1e3, "ms_per_gradient_k5" => tgrad/20*1e3,
    "grad_norm_at_ptrue_k5" => norm(g), "grad_all_finite" => all(isfinite, g),
    "N" => size(D.data,1), "Delta_t" => Δt, "delta_t" => δt, "S" => S, "gamma" => γ,
    "frozen_mod8_sha256" => depssha(MOD8)))
println("00 done")
