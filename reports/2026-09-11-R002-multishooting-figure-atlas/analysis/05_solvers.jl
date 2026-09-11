# 05 — different solvers inside the multiple-shooting loss.
# self-contained: frozen deps/ shadows the live repository tree
#
# Two stages.
#   (1) LOSS EVALUATION: the same multiple-shooting cost J(x0, p) is evaluated with
#       (a) the in-house FIXED-STEP Tsit5 used everywhere else in this report
#           (`ms_loss`, S sub-steps of δt = Δt/S per data interval), and
#       (b) the frozen mod 8 `forward_simulation_loss_windows` STIFF BRANCH
#           (`stiff_solver_=true`), which integrates each shooting window with a
#           DifferentialEquations.jl solver (`solver_`, `adaptive_`, `abs_tol_`, `rel_tol_`).
#       Both normalise as data_loss/N + γ·Σ smoothl1(p)/Np, so the numbers are directly
#       comparable.  γ = 0 here so the reported J is pure data misfit.
#       REFERENCE: Tsit5() adaptive with abstol = reltol = 1e-12 (same stiff branch).
#   (2) SWEEP: the guess-propagation sweep (mode=:propagate, NelderMead, 1000 iterations,
#       schedule [1,2,5,10,25,50,100], γ = 5e-2) run with the loss evaluated by
#       (a) in-house Tsit5 S=10 (via `run_sweep`), (b) DifferentialEquations Tsit5()
#       adaptive 1e-8, (c) Rosenbrock23() adaptive 1e-8.  (b)/(c) use a local loop that
#       mirrors `run_sweep`'s propagate logic exactly (same seed vector, same optimiser,
#       same options, minimiser copied into the next start).
#
# Outputs: results/solvers_eval.csv, results/solvers_sweep.csv, results/solvers_meta.json
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))
using DifferentialEquations, Logging, Statistics

# ---- frozen mod 8 (post-fix), loaded verbatim by line range ---------------------
run_block(depspath(MOD8), MOD8_ODEFUN...)
run_block(depspath(MOD8), MOD8_ODEFUN_POLY...)
run_block(depspath(MOD8), MOD8_PARAMS...)
run_block(depspath(MOD8), MOD8_DATA...)          # defines `data`, `Δt`
run_block(depspath(MOD8), MOD8_SMOOTHL1...)
run_block(depspath(MOD8), MOD8_FSL...)
run_block(depspath(MOD8), MOD8_FSLW...)          # forward_simulation_loss_windows
odefun_generic = (du,u,p,t) -> odefun_poly!(du,u,p,t; indices=multi_index_set, deg=3)

rhs!  = make_rhs(FHN_LIB)
D     = make_dataset(rhs!, FHN_P, [1.0, 1.0])
@assert maximum(abs.(D.data .- data)) < 1e-12   # core dataset == frozen mod 8 dataset
const DAT   = data                               # N×3, column 1 = time
const N_DAT = size(DAT, 1)
const X0    = DAT[1, 2:3]
const T0    = 0.0
const GAM_EVAL = 0.0
const S_REF = 10
const DT_REF = Δt / S_REF                        # 0.1

# ---- parameter vectors ---------------------------------------------------------
params = Pair{String,Vector{Float64}}[]
push!(params, "ptrue" => copy(FHN_P))
for k in 1:2
    push!(params, "ptrue_plus_0.02randn_seed$(k)" => FHN_P .+ 0.02 .* randn(MersenneTwister(k), 20))
end
@assert maximum(abs.(FHN_P .- fhn_p)) == 0.0     # library order == frozen mod 8 fhn_p

kappas = [1, 5, 25, 100]

# ---- helpers -------------------------------------------------------------------
_clean(s) = replace(string(s), ',' => ';', '\n' => ' ', '\r' => ' ')

"median wall time (ms) of `n` evaluations of f after one warm-up call"
function timed_median(f; n=5)
    f()
    ts = Float64[]
    for _ in 1:n; push!(ts, @elapsed f()); end
    return median(ts) * 1e3
end

# in-house fixed-step Tsit5 loss (γ = 0), S sub-steps of δt = Δt/S
inhouse_loss(p, κ, S; rhsf=rhs!) = ms_loss(X0, p, rhsf, DAT, Δt/S, S, GAM_EVAL, κ)

# frozen mod 8 stiff branch
function stiff_loss(p, κ; solver, tol, adaptive, δt=DT_REF, S=S_REF, γ=GAM_EVAL, odef=odefun_generic)
    forward_simulation_loss_windows(X0, [], p, odef, DAT, T0, δt, S, γ, κ, true;
        solver_=solver, abs_tol_=tol, rel_tol_=tol, adaptive_=adaptive, silent_=true)
end

# RHS-evaluation counter (one untimed extra evaluation; never inside a timed block)
function count_rhs_inhouse(p, κ, S)
    c = Ref(0)
    crhs!(du,u,pp,t) = (c[] += 1; rhs!(du,u,pp,t))
    try; inhouse_loss(p, κ, S; rhsf=crhs!); catch; end
    return c[]
end
function count_rhs_stiff(p, κ; kw...)
    c = Ref(0)
    cod(du,u,pp,t) = (c[] += 1; odefun_generic(du,u,pp,t))
    try; stiff_loss(p, κ; odef=cod, kw...); catch; end
    return c[]
end

# ---- method table --------------------------------------------------------------
# each entry: (method, setting, kind, kwargs)
inhouse_S = [1, 2, 5, 10, 20, 50]
stiff_methods = [
    ("diffeq_Tsit5_adaptive",         "abstol=reltol=1e-4", (solver=Tsit5(),         tol=1e-4,  adaptive=true,  δt=DT_REF)),
    ("diffeq_Tsit5_adaptive",         "abstol=reltol=1e-8", (solver=Tsit5(),         tol=1e-8,  adaptive=true,  δt=DT_REF)),
    ("diffeq_Rosenbrock23_adaptive",  "abstol=reltol=1e-8", (solver=Rosenbrock23(),  tol=1e-8,  adaptive=true,  δt=DT_REF)),
    ("diffeq_Rodas5_adaptive",        "abstol=reltol=1e-8", (solver=Rodas5(),        tol=1e-8,  adaptive=true,  δt=DT_REF)),
    ("diffeq_RadauIIA5_adaptive",     "abstol=reltol=1e-8", (solver=RadauIIA5(),     tol=1e-8,  adaptive=true,  δt=DT_REF)),
    ("diffeq_ImplicitEuler_fixed",    "dt=0.1",             (solver=ImplicitEuler(), tol=1e-8,  adaptive=false, δt=DT_REF)),
    ("diffeq_Tsit5_fixed",            "dt=0.1",             (solver=Tsit5(),         tol=1e-8,  adaptive=false, δt=DT_REF)),
]
const REF_METHOD  = "reference_diffeq_Tsit5_adaptive"
const REF_SETTING = "abstol=reltol=1e-12"
ref_kw = (solver=Tsit5(), tol=1e-12, adaptive=true, δt=DT_REF)

# ================================================================================
# stage 1 — loss evaluation
# ================================================================================
t_stage1 = time()
eval_rows = NamedTuple[]
for (pname, p) in params, κ in kappas
    # reference first
    Jref = NaN; ref_note = ""
    tref = NaN
    try
        Jref = stiff_loss(p, κ; ref_kw...)
        tref = timed_median(() -> stiff_loss(p, κ; ref_kw...))
    catch e
        ref_note = _clean(sprint(showerror, e))
    end
    nref = isfinite(Jref) ? count_rhs_stiff(p, κ; ref_kw...) : 0
    push!(eval_rows, (param=pname, window_size=κ, method=REF_METHOD, setting=REF_SETTING,
                      J=Jref, abs_err_vs_reference=(isfinite(Jref) ? 0.0 : NaN),
                      time_ms=tref, nfev=nref, note=ref_note))

    for S in inhouse_S
        J = NaN; note = ""; tms = NaN; nf = 0
        try
            J = inhouse_loss(p, κ, S)
            tms = timed_median(() -> inhouse_loss(p, κ, S))
            nf = count_rhs_inhouse(p, κ, S)
        catch e
            note = _clean(sprint(showerror, e))
        end
        push!(eval_rows, (param=pname, window_size=κ, method="inhouse_Tsit5_fixed",
                          setting="S=$S", J=J,
                          abs_err_vs_reference=(isfinite(J) && isfinite(Jref) ? abs(J-Jref) : NaN),
                          time_ms=tms, nfev=nf, note=note))
    end

    for (mname, sname, kw) in stiff_methods
        J = NaN; note = ""; tms = NaN; nf = 0
        try
            J = stiff_loss(p, κ; kw...)
            tms = timed_median(() -> stiff_loss(p, κ; kw...))
            nf = count_rhs_stiff(p, κ; kw...)
        catch e
            note = _clean(sprint(showerror, e))
        end
        push!(eval_rows, (param=pname, window_size=κ, method=mname, setting=sname, J=J,
                          abs_err_vs_reference=(isfinite(J) && isfinite(Jref) ? abs(J-Jref) : NaN),
                          time_ms=tms, nfev=nf, note=note))
    end
    println("eval done: $pname κ=$κ")
    flush(stdout)
end
write_csv(joinpath(RESULTS, "solvers_eval.csv"), eval_rows)
stage1_s = time() - t_stage1
n_fail = count(r -> !isempty(r.note), eval_rows)
println("stage 1 done in $(round(stage1_s, digits=1)) s; $(length(eval_rows)) rows, $n_fail failures")

# ================================================================================
# stage 2 — guess-propagation sweep under three loss evaluators
# ================================================================================
schedule = [1, 2, 5, 10, 25, 50, 100]
seeds    = [1, 2]
GAM_SWEEP = 5e-2
ITERS_PLAN = 1000

# budget estimate from stage-1 timings (median over κ at ptrue), 1500 f-evals/κ
med_ms(m, s) = begin
    v = [r.time_ms for r in eval_rows if r.method == m && r.setting == s && isfinite(r.time_ms)]
    isempty(v) ? NaN : median(v)
end
est_ms = Dict("diffeq_Tsit5_adaptive" => med_ms("diffeq_Tsit5_adaptive", "abstol=reltol=1e-8"),
              "diffeq_Rosenbrock23_adaptive" => med_ms("diffeq_Rosenbrock23_adaptive", "abstol=reltol=1e-8"))
est_total_s = sum(v -> isfinite(v) ? v : 10.0, values(est_ms)) * 1500 * length(schedule) * length(seeds) / 1e3
println("estimated stage-2 stiff sweep wall time: $(round(est_total_s, digits=0)) s")

reductions = String[]
iters = ITERS_PLAN
if est_total_s > 1500          # > 25 min of the ~40 min budget
    iters = 500
    push!(reductions, "iters reduced 1000 -> 500 for the DifferentialEquations arms (estimate $(round(est_total_s)) s > 1500 s)")
end
if est_total_s > 4000
    seeds = [1]
    push!(reductions, "seeds reduced to [1] for the DifferentialEquations arms")
end

sweep_rows = NamedTuple[]

# (a) in-house Tsit5 S=10 via run_sweep (always the planned 1000 iterations)
for seed in seeds
    r = run_sweep(rhs!, DAT, FHN_P, schedule; seed=seed, mode=:propagate, optimizer=:nm,
                  iters=ITERS_PLAN, γ=GAM_SWEEP, S=S_REF, penalty=:flat, x0_scale=0.1)
    for row in r.rows
        push!(sweep_rows, merge((method="inhouse_Tsit5_fixed_S10", setting="S=10",
                                 seed=seed, iters=ITERS_PLAN), row))
    end
    println("sweep (a) seed=$seed done, final p_err=$(r.rows[end].p_err)")
    flush(stdout)
end

# (b)/(c) DifferentialEquations arms — local loop mirroring run_sweep's :propagate logic
function stiff_sweep(seed, schedule; solver, tol, iters)
    d = 2; Np = 20
    z_seed = [DAT[1, 2:end]; 0.1 * randn(MersenneTwister(seed), Np)]
    z = copy(z_seed)
    rows = NamedTuple[]
    for κ in schedule
        f = z -> forward_simulation_loss_windows(z[1:d], [], z[d+1:end], odefun_generic, DAT,
                    T0, DT_REF, S_REF, GAM_SWEEP, κ, true;
                    solver_=solver, abs_tol_=tol, rel_tol_=tol, adaptive_=true, silent_=true)
        opts = Optim.Options(iterations=iters, g_tol=1e-8, store_trace=false, show_trace=false)
        t0 = time()
        res = with_logger(NullLogger()) do
            Optim.optimize(f, z, NelderMead(), opts)
        end
        wall = time() - t0
        m = Optim.minimizer(res); J = Optim.minimum(res); p = m[d+1:end]
        sc = recovery_score(p, FHN_P)
        push!(rows, (window_size=κ, J=J, p_err=norm(p - FHN_P), score=sc.score,
                     support_recall=sc.support_recall, false_positives=sc.false_positives,
                     wall_s=wall, f_calls=Optim.f_calls(res), iterations=Optim.iterations(res),
                     converged=Optim.converged(res), blowup=(J ≥ 1e3),
                     seed_p_err=norm(z_seed[d+1:end] - FHN_P)))
        copyto!(z, m)               # :propagate
    end
    return rows
end

for (mname, sname, solver) in (("diffeq_Tsit5_adaptive", "abstol=reltol=1e-8", Tsit5()),
                               ("diffeq_Rosenbrock23_adaptive", "abstol=reltol=1e-8", Rosenbrock23()))
    for seed in seeds
        rows = stiff_sweep(seed, schedule; solver=solver, tol=1e-8, iters=iters)
        for row in rows
            push!(sweep_rows, merge((method=mname, setting=sname, seed=seed, iters=iters), row))
        end
        println("sweep $mname seed=$seed done, final p_err=$(rows[end].p_err), wall=$(round(sum(r.wall_s for r in rows), digits=1)) s")
        flush(stdout)
    end
end
write_csv(joinpath(RESULTS, "solvers_sweep.csv"), sweep_rows)
stage2_s = time() - t_stage1 - stage1_s
println("stage 2 done in $(round(stage2_s, digits=1)) s")

# ================================================================================
# meta
# ================================================================================
tim = Dict{String,Any}()
for r in eval_rows
    isfinite(r.time_ms) || continue
    tim["$(r.method)|$(r.setting)|kappa=$(r.window_size)|$(r.param)"] = r.time_ms
end
meta = Dict{String,Any}(
  "script" => "analysis/05_solvers.jl",
  "frozen_mod8_sha256" => depssha(MOD8),
  "julia_version" => string(VERSION),
  "N" => N_DAT, "Delta_t" => Δt, "delta_t" => DT_REF, "S_reference" => S_REF,
  "x0" => X0, "t0" => T0,
  "gamma_eval" => GAM_EVAL, "gamma_sweep" => GAM_SWEEP,
  "kappas_eval" => kappas,
  "param_vectors" => [first(pp) for pp in params],
  "param_perturbation" => "p = p_true + 0.02*randn(MersenneTwister(k), 20), k = 1,2",
  "inhouse_S_grid" => inhouse_S,
  "inhouse_description" => "ms_loss: fixed-step Tsit5 (frozen deps/integrator.jl), S sub-steps of dt = Delta_t/S per data interval; loss = data_loss/N + gamma*sum(smoothl1(p))/Np",
  "stiff_branch_description" => "frozen mod 8 forward_simulation_loss_windows(..., stiff_solver_=true): each shooting window solved with DifferentialEquations.jl solver_, saveat = t_init:Delta_t:t_final, abstol=reltol=tol, adaptive_=adaptive, dt=delta_t; same normalisation data_loss/N + gamma*sum(smoothl1(p))/Np, so values are directly comparable with ms_loss",
  "stiff_methods" => ["$(m)|$(s)" for (m,s,_) in stiff_methods],
  "reference_solution" => "$(REF_METHOD) ($(REF_SETTING)): frozen mod 8 stiff branch, Tsit5(), adaptive_=true, abstol=reltol=1e-12; abs_err_vs_reference = |J_method - J_reference| at the same (param, window_size)",
  "eval_rows" => length(eval_rows),
  "eval_failures" => n_fail,
  "eval_failure_list" => [ "$(r.param)|$(r.method)|$(r.setting)|kappa=$(r.window_size): $(r.note)" for r in eval_rows if !isempty(r.note) ],
  "sweep_schedule" => schedule,
  "sweep_mode" => "propagate", "sweep_optimizer" => "NelderMead",
  "sweep_iters_inhouse" => ITERS_PLAN, "sweep_iters_diffeq" => iters,
  "sweep_seeds" => seeds,
  "sweep_seed_vector" => "[data[1,2:3]; 0.1*randn(MersenneTwister(seed), 20)]",
  "sweep_estimate_s" => est_total_s,
  "reductions" => isempty(reductions) ? ["none"] : reductions,
  "stage1_wall_s" => stage1_s, "stage2_wall_s" => stage2_s,
  "eval_time_ms" => tim,
)
write_json(joinpath(RESULTS, "solvers_meta.json"), meta)
println("05 done")
