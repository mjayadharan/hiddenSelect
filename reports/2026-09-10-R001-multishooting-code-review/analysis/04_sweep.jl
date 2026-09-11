# 04 — the headline guess-propagation sweep (report section 5).
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl"))
using DifferentialEquations, LinearAlgebra, Statistics, Optim, Random, Logging

C = load_pipeline!()
const WS = [1, 2, 5, 10, 25, 50, 100]
const ITERS = 2500
opts = Optim.Options(show_trace=false, iterations=ITERS)

Random.seed!(1)
x0_seed = [C.data[1:2]; 0.1*randn(20)]
seed_err = norm(x0_seed[3:end] - C.fhn_p)

lossfun(up, ws) = x -> forward_simulation_loss_windows(view(x,1:2), up, view(x,3:length(x)),
                         odefun, C.data, 0.0, C.δt, C.S, C.γ2, ws, false; silent_=true)

rows = NamedTuple[]
x0 = copy(x0_seed)                                   # ARM A: guess propagation
for ws in WS
    r = Optim.optimize(lossfun([], ws), x0, NelderMead(), opts)
    m = Optim.minimizer(r); copyto!(x0, m)
    push!(rows, (arm="guess_propagation", window_size=ws, J=Optim.minimum(r),
                 p_err=norm(m[3:end] - C.fhn_p)))
end
for ws in WS                                          # ARM B: no propagation
    r = Optim.optimize(lossfun([], ws), copy(x0_seed), NelderMead(), opts)
    m = Optim.minimizer(r)
    push!(rows, (arm="no_propagation", window_size=ws, J=Optim.minimum(r),
                 p_err=norm(m[3:end] - C.fhn_p)))
end
write_csv(joinpath(RESULTS, "sweep.csv"), rows)
write_json(joinpath(RESULTS, "sweep_meta.json"), Dict("optimiser" => "NelderMead", "iterations" => ITERS,
                        "opt_seed" => 1, "seed_p_err" => seed_err,
                        "window_sizes" => WS, "blowup_penalty" => 1000.0))
println("04 done")
