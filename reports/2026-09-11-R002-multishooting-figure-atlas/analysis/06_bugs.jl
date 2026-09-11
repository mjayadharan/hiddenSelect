# 06 — quantification of the repository defects reported in R001 (F1–F4, plus the
# monomial-ordering and global-`data` shadowing defects).
# self-contained: frozen deps/ shadows the live repository tree.
#
# Method: every defect is measured by EXECUTING the frozen PRE-FIX source
# (deps/prefix/…) next to the frozen POST-FIX source (deps/…).  The two versions
# of a function share a name, so each is loaded verbatim with `run_block` into its
# own module; every module carries its own `using` lines and its own copy of the
# frozen integrator.  Nothing is patched or re-typed here — the only difference
# between a `…Pre` and a `…Fix` module is which frozen file the lines came from.
#
# Outputs (analysis/results/):
#   bug_F2_w_residual.csv   bug_F2_slice.csv
#   bug_F1_jacobian.csv     bug_F1_meta.json
#   bug_F4_gradient.csv     bug_F4_bfgs.csv    bug_F4_bfgs_summary.json
#   bug_ordering.csv        bug_ordering_meta.json
#   bug_shadowing.csv       bug_summary.json

include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))
using DifferentialEquations, Logging, LinearAlgebra, Statistics, Random, ForwardDiff, Optim

# line ranges not in common.jl (identical in deps/ and deps/prefix/ for these blocks)
const MOD7_ODEFUN   = (19, 23)
const MOD7_SMOOTHL1 = (124, 131)

const INTEGRATOR = depspath("integrator.jl")

# ------------------------------------------------------------------ modules ----
# mod 7 `forward_simulation_loss_windows` — pre-fix (F2 present) and post-fix.
module M7Pre
    using LinearAlgebra, Statistics, Random, DifferentialEquations, Logging, ForwardDiff
end
module M7Fix
    using LinearAlgebra, Statistics, Random, DifferentialEquations, Logging, ForwardDiff
end
# mod 6 `odefun_new` — pre-fix (F1 linear truncation) and post-fix.
module M6Pre
    using LinearAlgebra, Statistics, Random, DifferentialEquations, Logging, ForwardDiff
end
module M6Fix
    using LinearAlgebra, Statistics, Random, DifferentialEquations, Logging, ForwardDiff
end
# mod 8 pre-fix old-order `odefun` (monomial-ordering defect).
module M8Ord
    using LinearAlgebra, Statistics, Random, DifferentialEquations, Logging, ForwardDiff
end
# mod 8 `forward_simulation_loss` — pre-fix (reads the global `data`) and post-fix.
module M8ShadPre
    using LinearAlgebra, Statistics, Random, DifferentialEquations, Logging, ForwardDiff
end
module M8ShadFix
    using LinearAlgebra, Statistics, Random, DifferentialEquations, Logging, ForwardDiff
end

for m in (M7Pre, M7Fix, M8ShadPre, M8ShadFix)
    Base.include(m, INTEGRATOR)                       # Tsit5Cache, integration_step!
end

run_block(prefixpath(MOD7), MOD7_ODEFUN...;        m=M7Pre)
run_block(prefixpath(MOD7), MOD7_SMOOTHL1...;      m=M7Pre)
run_block(prefixpath(MOD7), MOD7_FSLW_PREFIX...;   m=M7Pre)

run_block(depspath(MOD7),   MOD7_ODEFUN...;        m=M7Fix)
run_block(depspath(MOD7),   MOD7_SMOOTHL1...;      m=M7Fix)
run_block(depspath(MOD7),   MOD7_FSLW_FIXED...;    m=M7Fix)

run_block(prefixpath(MOD6), MOD6_ODEFUN_NEW_PREFIX...; m=M6Pre)
run_block(depspath(MOD6),   MOD6_ODEFUN_NEW_FIXED...;  m=M6Fix)

run_block(prefixpath(MOD8), MOD8_ODEFUN...;   m=M8Ord)

run_block(prefixpath(MOD8), MOD8_SMOOTHL1...; m=M8ShadPre)
run_block(prefixpath(MOD8), MOD8_FSL...;      m=M8ShadPre)
run_block(depspath(MOD8),   MOD8_SMOOTHL1...; m=M8ShadFix)
run_block(depspath(MOD8),   MOD8_FSL...;      m=M8ShadFix)

# ------------------------------------------------------------------- setup -----
rhs!  = make_rhs(FHN_LIB)
D     = make_dataset(rhs!, FHN_P, [1.0, 1.0])
data  = D.data                       # 101×3, column 1 = time
N     = size(data, 1)
Δt    = data[2,1] - data[1,1]
S     = 10
δt    = Δt / S                       # 0.1
γ     = 5e-2                         # mod 7 γ2
x0    = collect(data[1, 2:3])

# interleaved data vector [v1,w1,v2,w2,…] expected by the mod 7 loss
Dvec  = vec(permutedims(data[:, 2:3]))

# library order → pre-refactor monomial order (1,v,w,v²,vw,w²,v³,v²w,vw²,w³)
const PERM = old_to_new_perm()       # p_new = p_old[PERM] per 10-block
function to_old(p_new)
    p_old = similar(p_new)
    p_old[PERM]        .= p_new[1:10]
    p_old[10 .+ PERM]  .= p_new[11:20]
    return p_old
end
"Effective library-order vector that an OLD-order RHS reads out of the vector q."
function old_order_reads(q)
    return [q[PERM]; q[10 .+ PERM]]
end

quiet(f) = with_logger(NullLogger()) do; f(); end

# ==============================================================================
# (1) F2 — the stiff/DiffEq branch of mod 7 drops the whole w-residual
# ==============================================================================
# γ is set to 0 here so that `missing_fraction` measures the DATA loss only: the
# smooth-ℓ1 term is identical in both versions and would merely dilute the ratio.
const GAMMA_F2 = 0.0

fslw_pre(p_old, κ, stiff) = quiet() do
    M7Pre.forward_simulation_loss_windows(copy(x0), Any[], p_old, M7Pre.odefun, Dvec,
        0.0, δt, S, GAMMA_F2, κ, stiff;
        solver_=Tsit5(), adaptive_=true, abs_tol_=1e-8, rel_tol_=1e-8, silent_=true)
end
fslw_fix(p_old, κ, stiff) = quiet() do
    M7Fix.forward_simulation_loss_windows(copy(x0), Any[], p_old, M7Fix.odefun, Dvec,
        0.0, δt, S, GAMMA_F2, κ, stiff;
        solver_=Tsit5(), adaptive_=true, abs_tol_=1e-8, rel_tol_=1e-8, silent_=true)
end

F2_PARAMS = [("ptrue", copy(FHN_P))]
for k in 1:3
    push!(F2_PARAMS, ("pert$k", FHN_P .+ 0.05 .* randn(MersenneTwister(k), 20)))
end

f2rows = NamedTuple[]
for κ in (1, 5, 10, 25, 100), (nm, p_new) in F2_PARAMS
    p_old = to_old(p_new)
    Jpre = fslw_pre(p_old, κ, true)
    Jfix = fslw_fix(p_old, κ, true)
    Jexp = fslw_fix(p_old, κ, false)
    push!(f2rows, (window_size=κ, param=nm, J_prefix_stiff=Jpre, J_fixed_stiff=Jfix,
                   J_explicit=Jexp, missing_fraction=(Jfix - Jpre)/Jfix))
end
write_csv(joinpath(RESULTS, "bug_F2_w_residual.csv"), f2rows)
println("F2 table done ($(length(f2rows)) rows)")

# 1-D slice in the ẇ-equation coefficient of v (library index 15, true 0.08)
f2slice = NamedTuple[]
for p15 in range(-0.5, 0.6; length=61)
    p_new = copy(FHN_P); p_new[15] = p15
    p_old = to_old(p_new)
    push!(f2slice, (p15=p15, J_prefix_stiff=fslw_pre(p_old, 10, true),
                    J_fixed_stiff=fslw_fix(p_old, 10, true)))
end
write_csv(joinpath(RESULTS, "bug_F2_slice.csv"), f2slice)
println("F2 slice done ($(length(f2slice)) rows)")

# ==============================================================================
# (2) F1 — mod 6 `odefun_new` is a linear truncation of the vector field
# ==============================================================================
P_OLD = to_old(FHN_P)
rhs_pre(y)   = M6Pre.odefun_new(y, P_OLD)
rhs_fix(y)   = M6Fix.odefun_new(y, P_OLD)
rhs_exact(y) = rhs!(zeros(eltype(y), 2), y, FHN_P, 0.0)

jac(f, u) = ForwardDiff.jacobian(f, u)
function eig2(J)
    ev = ComplexF64.(eigvals(J))
    sort!(ev, by = z -> (-real(z), -imag(z)))
    return ev
end

f1rows = NamedTuple[]
maxfix = 0.0
idxs = 1:25:size(D.alldata, 1)
for i in idxs
    u  = collect(D.alldata[i, :])
    t  = D.tsall[i]
    Jp = jac(rhs_pre, u); Jf = jac(rhs_fix, u); Je = jac(rhs_exact, u)
    global maxfix = max(maxfix, maximum(abs.(Jf .- Je)))
    ep = eig2(Jp); ef = eig2(Jf)
    push!(f1rows, (t=t, v=u[1], w=u[2],
        J11_prefix=Jp[1,1], J12_prefix=Jp[1,2], J21_prefix=Jp[2,1], J22_prefix=Jp[2,2],
        J11_fixed=Jf[1,1],  J12_fixed=Jf[1,2],  J21_fixed=Jf[2,1],  J22_fixed=Jf[2,2],
        J11_exact=Je[1,1],
        eig_re1_prefix=real(ep[1]), eig_im1_prefix=imag(ep[1]),
        eig_re2_prefix=real(ep[2]), eig_im2_prefix=imag(ep[2]),
        eig_re1_fixed=real(ef[1]),  eig_im1_fixed=imag(ef[1]),
        eig_re2_fixed=real(ef[2]),  eig_im2_fixed=imag(ef[2]),
        trace_prefix=tr(Jp), trace_fixed=tr(Jf),
        det_prefix=det(Jp),  det_fixed=det(Jf)))
end
write_csv(joinpath(RESULTS, "bug_F1_jacobian.csv"), f1rows)

u_ref = [1.5, 0.4]
Jp_ref = jac(rhs_pre, u_ref); Jf_ref = jac(rhs_fix, u_ref); Je_ref = jac(rhs_exact, u_ref)
write_json(joinpath(RESULTS, "bug_F1_meta.json"), Dict(
    "u_ref"               => u_ref,
    "rhs_prefix_at_uref"  => collect(Float64.(rhs_pre(u_ref))),
    "rhs_fixed_at_uref"   => collect(Float64.(rhs_fix(u_ref))),
    "rhs_exact_at_uref"   => collect(Float64.(rhs_exact(u_ref))),
    "J11_prefix_at_uref"  => Jp_ref[1,1],
    "J11_fixed_at_uref"   => Jf_ref[1,1],
    "J11_exact_at_uref"   => Je_ref[1,1],
    "jac_prefix_at_uref"  => vec(Jp_ref),
    "jac_fixed_at_uref"   => vec(Jf_ref),
    "jac_exact_at_uref"   => vec(Je_ref),
    "max_abs_fixed_minus_exact_over_trajectory" => maxfix,
    "n_trajectory_points" => length(idxs),
    "mod6_prefix_sha256"  => prefixsha(MOD6),
    "mod6_fixed_sha256"   => depssha(MOD6)))
println("F1 done ($(length(f1rows)) trajectory points, max|fixed-exact| = $maxfix)")

# ==============================================================================
# (3) F4 — the sweep's "gradient" differentiates the single-shooting loss
# ==============================================================================
z0 = [collect(data[1, 2:3]); FHN_P .+ 0.05 .* randn(MersenneTwister(4), 20)]
f_single = make_objective(rhs!, data, δt, S, γ, N - 1)          # κ = 100
g_single = ForwardDiff.gradient(f_single, z0)

f4rows = NamedTuple[]
for κ in (1, 2, 5, 10, 25, 50, 100)
    f_win = make_objective(rhs!, data, δt, S, γ, κ)
    g_win = ForwardDiff.gradient(f_win, z0)
    cs    = dot(g_win, g_single) / (norm(g_win) * norm(g_single))
    push!(f4rows, (window_size=κ, grad_norm_window=norm(g_win),
                   grad_norm_single=norm(g_single),
                   cosine_similarity=cs, angle_deg=acosd(clamp(cs, -1.0, 1.0)),
                   rel_diff=norm(g_win .- g_single)/norm(g_win)))
end
write_csv(joinpath(RESULTS, "bug_F4_gradient.csv"), f4rows)
println("F4 gradient table done")

# BFGS on the κ=5 windowed objective with the correct vs the wrong gradient
f_w5  = make_objective(rhs!, data, δt, S, γ, 5)
cfg_w = ForwardDiff.GradientConfig(f_w5,     z0, ForwardDiff.Chunk{12}())
cfg_s = ForwardDiff.GradientConfig(f_single, z0, ForwardDiff.Chunk{12}())
gw!(g, x) = ForwardDiff.gradient!(g, f_w5,     x, cfg_w)
gs!(g, x) = ForwardDiff.gradient!(g, f_single, x, cfg_s)
opts = Optim.Options(iterations=300, store_trace=true, extended_trace=false, g_tol=1e-8)

bfgs_rows = NamedTuple[]
bfgs_meta = Dict{String,Any}()
for (variant, g!) in (("correct", gw!), ("wrong", gs!))
    t0 = time()
    try
        res  = Optim.optimize(f_w5, g!, copy(z0), BFGS(), opts)
        wall = time() - t0
        for tr_ in Optim.trace(res)
            push!(bfgs_rows, (variant=variant, iteration=tr_.iteration, J_window=tr_.value))
        end
        m = Optim.minimizer(res)
        bfgs_meta["$(variant)_p_err"]      = norm(m[3:end] .- FHN_P)
        bfgs_meta["$(variant)_J_final"]    = Optim.minimum(res)
        bfgs_meta["$(variant)_iterations"] = Optim.iterations(res)
        bfgs_meta["$(variant)_f_calls"]    = Optim.f_calls(res)
        bfgs_meta["$(variant)_g_calls"]    = Optim.g_calls(res)
        bfgs_meta["$(variant)_converged"]  = Optim.converged(res)
        bfgs_meta["$(variant)_wall_s"]     = wall
        bfgs_meta["$(variant)_error"]      = nothing
    catch e
        wall = time() - t0
        msg  = first(sprint(showerror, e), 400)
        bfgs_meta["$(variant)_p_err"]      = NaN
        bfgs_meta["$(variant)_J_final"]    = NaN
        bfgs_meta["$(variant)_iterations"] = -1
        bfgs_meta["$(variant)_f_calls"]    = -1
        bfgs_meta["$(variant)_g_calls"]    = -1
        bfgs_meta["$(variant)_converged"]  = false
        bfgs_meta["$(variant)_wall_s"]     = wall
        bfgs_meta["$(variant)_error"]      = msg
        println("  BFGS variant=$variant threw: $msg")
    end
end
write_csv(joinpath(RESULTS, "bug_F4_bfgs.csv"), bfgs_rows)
bfgs_meta["J_window_at_start"] = f_w5(z0)
bfgs_meta["p_err_at_start"]    = norm(z0[3:end] .- FHN_P)
bfgs_meta["kappa_window"]      = 5
bfgs_meta["kappa_single"]      = N - 1
write_json(joinpath(RESULTS, "bug_F4_bfgs_summary.json"), bfgs_meta)
println("F4 BFGS done ($(length(bfgs_rows)) trace rows)")

# ==============================================================================
# (4) monomial-ordering defect: the pre-fix mod 8 sweep optimised the OLD-order
#     `odefun` while the data and the polish/plots used the library order
# ==============================================================================
new_labels = [(j <= 10 ? "dv/dt:" : "dw/dt:") * FHN_LIB.labels[((j-1) % 10) + 1] for j in 1:20]
old_labels = [(j <= 10 ? "dv/dt:" : "dw/dt:") * OLD_ORDER_LABELS[((j-1) % 10) + 1] for j in 1:20]
p_eff = old_order_reads(FHN_P)
ordrows = [(index=j, old_label=old_labels[j], new_label=new_labels[j],
            p_true_new_order=FHN_P[j], value_old_order_reads=p_eff[j]) for j in 1:20]
write_csv(joinpath(RESULTS, "bug_ordering.csv"), ordrows)

odefun_old = (du, u, p, t) -> M8Ord.odefun(du, u, p, t)
ordmeta = Dict{String,Any}(
    "spurious_p_err"     => norm(FHN_P .- p_eff),
    "p_true_norm"        => norm(FHN_P),
    "mod8_prefix_sha256" => prefixsha(MOD8))
for κ in (1, 5, 100)
    ordmeta["J_correct_order_k$(κ)"] = ms_loss(x0, FHN_P, rhs!,       data, δt, S, γ, κ)
    ordmeta["J_old_order_k$(κ)"]     = ms_loss(x0, FHN_P, odefun_old, data, δt, S, γ, κ)
end
write_json(joinpath(RESULTS, "bug_ordering_meta.json"), ordmeta)
println("ordering done (spurious p_err = $(ordmeta["spurious_p_err"]))")

# ==============================================================================
# (5) global-`data` shadowing in the pre-fix mod 8 `forward_simulation_loss`
# ==============================================================================
data_noisy = copy(data)
data_clean = copy(D.clean)
data_half  = copy(data); data_half[:, 2:3] .*= 0.5
Core.eval(M8ShadPre, :(data = $(data_noisy)))
Core.eval(M8ShadFix, :(data = $(data_noisy)))

shadrows = NamedTuple[]
for (vname, mod) in (("prefix", M8ShadPre), ("fixed", M8ShadFix)),
    (dname, dmat) in (("noisy", data_noisy), ("clean", data_clean), ("half", data_half))
    J = quiet() do
        Base.invokelatest(getfield(mod, :forward_simulation_loss),
                          copy(x0), FHN_P, rhs!, dmat, 0.0, δt, S, γ, false)
    end
    push!(shadrows, (version=vname, dataset=dname, J=J))
end
write_csv(joinpath(RESULTS, "bug_shadowing.csv"), shadrows)
getJ(v, d) = shadrows[findfirst(r -> r.version == v && r.dataset == d, shadrows)].J
pre_identical = (getJ("prefix","noisy") == getJ("prefix","clean") == getJ("prefix","half"))
fix_differs   = (getJ("fixed","noisy") != getJ("fixed","clean") &&
                 getJ("fixed","noisy") != getJ("fixed","half") &&
                 getJ("fixed","clean") != getJ("fixed","half"))
println("shadowing done (prefix identical = $pre_identical, fixed differs = $fix_differs)")

# ==============================================================================
# (6) headline summary
# ==============================================================================
f2_at10 = f2rows[findfirst(r -> r.window_size == 10 && r.param == "ptrue", f2rows)]
f4_at5  = f4rows[findfirst(r -> r.window_size == 5, f4rows)]

summary = Dict{String,Any}(
    "F2_missing_fraction_k10_ptrue" => f2_at10.missing_fraction,
    "F2_J_prefix_stiff_k10_ptrue"   => f2_at10.J_prefix_stiff,
    "F2_J_fixed_stiff_k10_ptrue"    => f2_at10.J_fixed_stiff,
    "F2_J_explicit_k10_ptrue"       => f2_at10.J_explicit,
    "F2_gamma_used"                 => GAMMA_F2,
    "F1_J11_prefix_at_uref"         => Jp_ref[1,1],
    "F1_J11_fixed_at_uref"          => Jf_ref[1,1],
    "F1_J11_exact_at_uref"          => Je_ref[1,1],
    "F1_max_abs_fixed_minus_exact"  => maxfix,
    "F4_cosine_similarity_k5"       => f4_at5.cosine_similarity,
    "F4_angle_deg_k5"               => f4_at5.angle_deg,
    "F4_rel_diff_k5"                => f4_at5.rel_diff,
    "F4_bfgs_correct_p_err"         => bfgs_meta["correct_p_err"],
    "F4_bfgs_wrong_p_err"           => bfgs_meta["wrong_p_err"],
    "F4_bfgs_wrong_error"           => bfgs_meta["wrong_error"],
    "ordering_spurious_p_err"       => ordmeta["spurious_p_err"],
    "shadowing_prefix_identical"    => pre_identical,
    "shadowing_fixed_differs"       => fix_differs,
    "N" => N, "Delta_t" => Δt, "delta_t" => δt, "S" => S, "gamma" => γ,
    "sha256_mod6_prefix" => prefixsha(MOD6), "sha256_mod6_fixed" => depssha(MOD6),
    "sha256_mod7_prefix" => prefixsha(MOD7), "sha256_mod7_fixed" => depssha(MOD7),
    "sha256_mod8_prefix" => prefixsha(MOD8), "sha256_mod8_fixed" => depssha(MOD8),
    "sha256_integrator"  => depssha("integrator.jl"),
    "sha256_helper_functions" => depssha("helper_functions.jl"))
write_json(joinpath(RESULTS, "bug_summary.json"), summary)

println("\n================ 06_bugs summary ================")
println("F2  missing fraction κ=10, p*:  ", f2_at10.missing_fraction,
        "   (prefix ", f2_at10.J_prefix_stiff, " vs fixed ", f2_at10.J_fixed_stiff,
        ", explicit ", f2_at10.J_explicit, ")")
println("F1  J11 at u=(1.5,0.4):  prefix ", Jp_ref[1,1], "  fixed ", Jf_ref[1,1],
        "  exact ", Je_ref[1,1], "   max|fixed-exact| ", maxfix)
println("F4  cosine(g_window κ=5, g_single κ=100) = ", f4_at5.cosine_similarity,
        "  angle ", f4_at5.angle_deg, " deg")
println("F4  BFGS p_err: correct ", bfgs_meta["correct_p_err"],
        "  wrong ", bfgs_meta["wrong_p_err"],
        "  (wrong error: ", bfgs_meta["wrong_error"], ")")
println("ORD spurious p_err = ", ordmeta["spurious_p_err"],
        "   J(p*) correct/old at κ=5: ", ordmeta["J_correct_order_k5"], " / ",
        ordmeta["J_old_order_k5"])
println("SHD prefix identical across datasets = ", pre_identical,
        ",  fixed differs = ", fix_differs)
println("06 done")
