# hs_core.jl — R002 analysis core: a corrected, allocation-light, dimension-generic
# implementation of the repository's multiple-shooting loss, plus the guess-
# propagation sweep driver. It builds ONLY on the frozen integrator and multi-index
# helpers in deps/; gate G2 in verify_R002.py asserts that `ms_loss` reproduces the
# frozen mod 8 `forward_simulation_loss_windows` (explicit branch) to 1e-12.
#
# Conventions (report §2): data matrix is N×(1+d), column 1 = time; parameters are
# ordered by `multiindices(d, deg)` — for d=2, deg=3 that is
#   1, w, w², w³, v, vw, vw², v², v²w, v³   (v = x₁, w = x₂), first the v̇ block then the ẇ block.
using LinearAlgebra, Statistics, Random, ForwardDiff, Optim

include(depspath("integrator.jl"))         # Tsit5Cache, integration_step!, integrate  (frozen)
include(depspath("helper_functions.jl"))   # multiindices, multiindex_mapping          (frozen)

# ------------------------------------------------------------------ library ----
struct PolyLib
    dim::Int
    deg::Int
    idx::Vector{Vector{Int}}      # exponent multi-indices, in multiindices() order
    labels::Vector{String}
end
function PolyLib(dim, deg; names=nothing)
    idx = multiindices(dim, deg)
    _, labels = multiindex_mapping(dim, deg; state_names=names)
    PolyLib(dim, deg, idx, labels)
end
nmono(L::PolyLib) = length(L.idx)
nparams(L::PolyLib) = L.dim * nmono(L)
"All parameter labels, e.g. `dv/dt: v^3`."
param_labels(L::PolyLib) = ["d$(n)/dt: $(m)" for n in (L.dim == 2 ? ["v","w"] : ["x$(i)" for i in 1:L.dim]) for m in L.labels]

@inline ipow(x, e::Int) = e == 1 ? x : e == 2 ? x*x : e == 3 ? x*x*x : x^e
"Return an in-place RHS `rhs!(du,u,p,t)` for the library (no allocation, AD-safe)."
function make_rhs(L::PolyLib)
    idx = [Tuple(a) for a in L.idx]
    nm = length(idx); dim = L.dim
    function rhs!(du, u, p, t)
        @inbounds for i in 1:dim
            off = (i-1)*nm
            s = zero(promote_type(eltype(u), eltype(p)))
            for k in 1:nm
                α = idx[k]
                m = one(eltype(u))
                for j in 1:dim
                    e = α[j]
                    e == 0 || (m *= ipow(u[j], e))
                end
                s += p[off+k]*m
            end
            du[i] = s
        end
        return du
    end
    return rhs!
end

# FHN in the library ordering (identical to frozen mod 8 `fhn_p`)
const FHN_LIB = PolyLib(2, 3; names=["v","w"])
const FHN_P   = [0.5, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1/3,
                 0.7/12.5, -0.8/12.5, 0.0, 0.0, 1.0/12.5, 0.0, 0.0, 0.0, 0.0, 0.0]
# The pre-refactor (mod 7 `odefun`) ordering: 1, v, w, v², vw, w², v³, v²w, vw², w³
const OLD_ORDER_LABELS = ["1","v","w","v^2","v w","w^2","v^3","v^2 w","v w^2","w^3"]
"Permutation π with p_new = p_old[π] (per state block)."
function old_to_new_perm(L::PolyLib=FHN_LIB)
    old = Dict("1"=>(0,0),"v"=>(1,0),"w"=>(0,1),"v^2"=>(2,0),"v w"=>(1,1),"w^2"=>(0,2),
               "v^3"=>(3,0),"v^2 w"=>(2,1),"v w^2"=>(1,2),"w^3"=>(0,3))
    oldidx = [old[l] for l in OLD_ORDER_LABELS]
    [findfirst(==(Tuple(a)), oldidx) for a in L.idx]
end

# ------------------------------------------------------------------ data ------
"""
    make_dataset(rhs!, p, y0; T_end, δt_fine, downsample, crop, noise_rel, seed)
Integrate with the frozen fixed-step Tsit5 at δt_fine, downsample, crop, add
per-component relative Gaussian noise. Returns (data N×(1+d), fine t, fine X, clean N×d).
Defaults reproduce the repository's FHN recipe exactly (mod 8 lines 103–157).
"""
function make_dataset(rhs!, p, y0; T_end=156.0, δt_fine=0.01, downsample=100, crop=(50.0,150.0),
                      noise_rel=0.05, seed=1287436679)
    cache = Tsit5Cache(copy(y0))
    Nst = Int(round(T_end/δt_fine))
    _, ys = integrate(rhs!, copy(y0), p, 0.0, Nst, δt_fine, cache)
    tsall = collect(0.0:δt_fine:T_end)
    alldata = reduce(hcat, ys)'                      # (Nst+1)×d
    data = alldata[1:downsample:end, :]
    ts = tsall[1:downsample:end]
    i1 = argmin(abs.(tsall .- crop[1])); i2 = argmin(abs.(tsall .- crop[2]))
    alldata = alldata[i1:i2, :]; tsall = tsall[i1:i2]
    j1 = argmin(abs.(ts .- crop[1])); j2 = argmin(abs.(ts .- crop[2]))
    ts = ts[j1:j2]; t1 = ts[1]; ts = ts .- t1; tsall = tsall .- t1
    data = Matrix(data[j1:j2, :])
    clean = copy(data)
    rng = Xoshiro(seed)        # == Random.seed!(seed) on Julia ≥1.7 (same Xoshiro256++ stream as mod 8)
    for j in 1:size(data,2)   # mod 8 draws v-noise then w-noise from one stream
        data[:, j] .+= noise_rel * std(data[:, j]) * randn(rng, size(data,1))
    end
    return (data=hcat(ts, data), tsall=tsall, alldata=Matrix(alldata), clean=hcat(ts, clean))
end

# ------------------------------------------------------------------ loss ------
"""
    ms_loss(x0, p, rhs!, data, δt, S, γ, κ; penalty=:flat, blow=1e3)
Multiple-shooting cost with window size κ (data intervals per window), S fixed
Tsit5 substeps of length δt per data interval, smooth-ℓ1 weight γ.
Every window restarts from the DATUM at its left end (no continuity constraint).
Normalisation follows frozen mod 8: data_loss/N + γ·Σ smoothl1(p_j)/Np.
penalty=:flat   → return 1e3 on blow-up (repository behaviour, a plateau);
penalty=:graded → return 1e3·(1 + (N-1-i)/(N-1)) + partial loss: blow-ups that
                  occur later are penalised less, giving a gradient-free optimiser a direction.
"""
function smoothl1(x, alpha=500)
    ax = alpha*x
    abs(ax) > 40 ? abs(x) : inv(alpha)*(log(1+exp(-alpha*x)) + log(1+exp(alpha*x)))
end
function ms_loss(x0, p, rhs!, data::AbstractMatrix, δt, S, γ, κ; penalty::Symbol=:flat, blow=1e3)
    N = size(data,1); d = size(data,2)-1
    T = promote_type(eltype(x0), eltype(p), Float64)
    cache = Tsit5Cache(Vector{T}(undef, d))
    x = cache.ycur
    loss = zero(T)
    @inbounds for j in 1:d
        x[j] = x0[j]
        loss += abs2(x0[j] - data[1, j+1])
    end
    t = data[1,1]
    @inbounds for i in 1:N-1
        if (i-1) % κ == 0
            for j in 1:d; x[j] = data[i, j+1]; end
        end
        for _ in 1:S
            integration_step!(cache, rhs!, x, t, p, δt, false)
            t += δt
        end
        bad = false
        for j in 1:d
            (abs(x[j]) > blow || isnan(x[j])) && (bad = true)
        end
        if bad
            penalty == :flat && return T(blow)
            return T(blow)*(1 + (N-1-i)/(N-1)) + loss/N
        end
        for j in 1:d; loss += abs2(x[j] - data[i+1, j+1]); end
    end
    sp = zero(T)
    for pj in p; sp += γ*smoothl1(pj); end
    return loss/N + sp/length(p)
end

"Closure over the packed vector z = [x0; p]."
function make_objective(rhs!, data, δt, S, γ, κ; d=size(data,2)-1, penalty=:flat)
    z -> ms_loss(view(z, 1:d), view(z, d+1:length(z)), rhs!, data, δt, S, γ, κ; penalty=penalty)
end

"Integrate the model from x0 with parameters p on the fine grid (for plotting fits)."
function simulate(rhs!, x0, p, t_end, δt)
    cache = Tsit5Cache(copy(x0))
    Nst = Int(round(t_end/δt))
    ts, ys = integrate(rhs!, copy(x0), p, 0.0, Nst, δt, cache)
    return ts, reduce(hcat, ys)'
end

# --------------------------------------------------------------- metrics -------
"""
    recovery_score(p, p_true; tol_abs=0.05, tol_rel=0.25)
Fractional system-recovery score (report §2): per parameter, credit 1 if it is
correctly ZERO (|p_j| < tol_abs when p*_j = 0) or correctly NON-ZERO with
|p_j - p*_j| ≤ tol_rel·|p*_j|; the score is the mean credit over all Np entries.
Also returns the support-recovery flags.
"""
function recovery_score(p, pt; tol_abs=0.05, tol_rel=0.25)
    credit = 0.0; supp_ok = 0; n_supp = 0
    for j in eachindex(pt)
        if pt[j] == 0
            c = abs(p[j]) < tol_abs
            credit += c
        else
            n_supp += 1
            c = abs(p[j]-pt[j]) ≤ tol_rel*abs(pt[j])
            credit += c
            supp_ok += abs(p[j]) ≥ tol_abs
        end
    end
    return (score=credit/length(pt), support_recall=supp_ok/max(n_supp,1),
            false_positives=count(j -> pt[j]==0 && abs(p[j])≥tol_abs, eachindex(pt)))
end

# ----------------------------------------------------------------- sweep -------
"""
    run_sweep(rhs!, data, p_true, schedule; seed, mode, optimizer, iters, γ, S, δt, penalty, x0_scale)
Guess-propagation sweep over window sizes `schedule`.
mode ∈ (:propagate, :best, :reset); optimizer ∈ (:nm, :bfgs, :lbfgs).
Returns (rows, minimizers): one row per window size with cost, ‖p−p*‖, recovery
score, wall time, f-evaluations, iterations, converged flag, blow-up flag.
"""
function run_sweep(rhs!, data, p_true, schedule; seed=1, mode=:propagate, optimizer=:nm,
                   iters=2500, γ=5e-2, S=10, δt=nothing, penalty=:flat, x0_scale=0.1,
                   trace_every=0, g_tol=1e-8, p_init=nothing)
    d = size(data,2)-1; Np = length(p_true)
    Δt = data[2,1]-data[1,1]
    δt === nothing && (δt = Δt/S)
    rng = MersenneTwister(seed)
    z_seed = p_init === nothing ? [data[1, 2:end]; x0_scale*randn(rng, Np)] : [data[1, 2:end]; p_init]
    z = copy(z_seed); best = Inf
    rows = NamedTuple[]; mins = Dict{Int,Vector{Float64}}(); traces = Dict{Int,Vector{Float64}}()
    for κ in schedule
        f = make_objective(rhs!, data, δt, S, γ, κ; d=d, penalty=penalty)
        opts = Optim.Options(iterations=iters, g_tol=g_tol, store_trace=trace_every>0,
                             show_trace=false)
        t0 = time()
        if optimizer == :nm
            res = Optim.optimize(f, z, NelderMead(), opts)
        else
            cfg = ForwardDiff.GradientConfig(f, z, ForwardDiff.Chunk{min(12, length(z))}())
            g!(g, x) = ForwardDiff.gradient!(g, f, x, cfg)
            alg = optimizer == :bfgs ? BFGS() : LBFGS()
            res = Optim.optimize(f, g!, z, alg, opts)
        end
        wall = time()-t0
        m = Optim.minimizer(res); J = Optim.minimum(res)
        p = m[d+1:end]
        sc = recovery_score(p, p_true)
        push!(rows, (window_size=κ, J=J, p_err=norm(p-p_true), score=sc.score,
                     support_recall=sc.support_recall, false_positives=sc.false_positives,
                     wall_s=wall, f_calls=Optim.f_calls(res), iterations=Optim.iterations(res),
                     converged=Optim.converged(res), blowup=(J ≥ 1e3),
                     seed_p_err=norm(z_seed[d+1:end]-p_true)))
        mins[κ] = copy(m)
        trace_every > 0 && (traces[κ] = [t.value for t in Optim.trace(res)])
        if mode == :propagate
            copyto!(z, m)
        elseif mode == :best
            if J < best; best = J; copyto!(z, m); end
        else
            copyto!(z, z_seed)
        end
    end
    return (rows=rows, minimizers=mins, seed_vector=z_seed, traces=traces)
end

"Flatten a Dict{Int,Vector} of minimizers into long rows (window_size, index, value)."
function minimizer_rows(mins; extra=NamedTuple())
    out = NamedTuple[]
    for κ in sort(collect(keys(mins))), (j, v) in enumerate(mins[κ])
        push!(out, merge(extra, (window_size=κ, index=j, value=v)))
    end
    return out
end
