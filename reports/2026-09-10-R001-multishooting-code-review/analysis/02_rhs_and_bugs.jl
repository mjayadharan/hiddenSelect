# 02 — (A) odefun_poly! equivalence, (B) finding F1, (C)/(D) finding F3.
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl"))
using DifferentialEquations, LinearAlgebra, Random, ForwardDiff

include(depspath("integrator.jl")); include(depspath("helper_functions.jl"))
run_block(MOD7, MOD7_ODEFUN...)          # odefun        (old monomial order)
run_block(MOD8, MOD8_ODEFUN_POLY...)     # odefun_poly!  (new monomial order)
run_block(MOD6, MOD6_ODEFUN_NEW...)      # odefun_new    (finding F1)

p_old = [0.5,1.0,-1.0,0.0,0.0,0.0,-1/3,0.0,0.0,0.0, 0.7/12.5,1.0/12.5,-0.8/12.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
p_new = [0.5,-1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,-1/3, 0.7/12.5,-0.8/12.5,0.0,0.0,1.0/12.5,0.0,0.0,0.0,0.0,0.0]
mi = multiindices(2, 3)
_, monomial_names = multiindex_mapping(2, 3)

# --- (A) odefun_poly! (mod 8) must reproduce odefun (mod 7) --------------
Random.seed!(11); maxdiff = 0.0
for _ in 1:200
    u = randn(2) .* 2; a = zeros(2); b = zeros(2)
    odefun(a, u, p_old, 0.0); odefun_poly!(b, u, p_new, 0.0; indices=mi, deg=3)
    global maxdiff = max(maxdiff, maximum(abs.(a .- b)))
end

# --- (B) F1: odefun_new drops every nonlinear term ------------------------
u = [1.5, 0.4]
F_new = odefun_new(u, p_old)
F_ref = zeros(2); odefun(F_ref, u, p_old, 0.0)
F_lin = [p_old[1]+p_old[2]*u[1]+p_old[3]*u[2], p_old[11]+p_old[12]*u[1]+p_old[13]*u[2]]
J_new = ForwardDiff.jacobian(y -> odefun_new(y, p_old), u)
J_ref = ForwardDiff.jacobian(y -> (z = zeros(eltype(y), 2); odefun(z, y, p_old, 0.0); z), u)

# --- (C)/(D) F3: the two mod 8 run-time errors ----------------------------
"Error type, unwrapping the LoadError that include_string wraps around it."
function errname(f)
    try; f(); return "NO_ERROR"
    catch e
        e isa LoadError && (e = e.error)
        return string(nameof(typeof(e)))
    end
end
err_fhn_u0 = errname(() -> include_string(Main,
    "ODEProblem((du,u,p,t)->nothing, fhn_u0, (0.0,1.0), $(p_new))"))
run_block(MOD8, MOD8_FSL...)             # mod 8 forward_simulation_loss
# The driver always has a flat 2N-vector `data` in scope. mod 8's body reads that
# GLOBAL `data` (`d, N = size(data)`) and ignores its own `data_` argument, so the
# global must exist to reproduce the driver's real failure mode (BoundsError on a
# 1-tuple) rather than a bare UndefVarError. Recorded separately as D2 below.
data = randn(202)
err_fsl = errname(() -> forward_simulation_loss([data[1],data[2]], p_old, odefun,
                                                data, 0.0, 0.1, 10, 0.0, false))
reads_global_data = occursin("size(data)", join(depslines(MOD8)[MOD8_FSL[1]:MOD8_FSL[2]], "
"))

write_json(joinpath(RESULTS, "rhs_and_bugs.json"), Dict(
        "A_odefun_poly_maxabsdiff" => maxdiff,
        "A_monomial_order_mod8"    => monomial_names,
        "B_u"                      => u,
        "B_odefun_new"             => F_new,
        "B_odefun_true"            => F_ref,
        "B_linear_truncation"      => F_lin,
        "B_jac_odefun_new"         => [J_new[1,1], J_new[1,2], J_new[2,1], J_new[2,2]],
        "B_jac_true"               => [J_ref[1,1], J_ref[1,2], J_ref[2,1], J_ref[2,2]],
        "C_mod8_fhn_u0_error"      => err_fhn_u0,
        "D_mod8_fsl_error"         => err_fsl,
        "D2_mod8_fsl_reads_global_data" => reads_global_data,
        "frozen_mod6_sha256"       => depssha(MOD6),
        "frozen_mod8_sha256"       => depssha(MOD8)))
println("02 done")
