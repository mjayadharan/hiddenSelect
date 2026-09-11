# 03 — cost landscapes: 2-D slices, 1-D random slices (local-minima counts),
#      Hessian spectrum at p* vs κ, J_κ(p*) vs κ, and single- vs multiple-shooting
#      sensitivity of the cost to the initial condition.
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))

rhs! = make_rhs(FHN_LIB)
D = make_dataset(rhs!, FHN_P, [1.0, 1.0])
data = D.data; N = size(data,1); Δt = data[2,1]-data[1,1]; S = 10; δt = Δt/S; γ = 5e-2
x0 = data[1, 2:3]
J(p, κ; g=γ) = ms_loss(x0, p, rhs!, data, δt, S, g, κ)

# ---- 2-D slices in the (p_v, p_v³) plane of the v̇ equation (indices 5 and 10) ----
a_range = range(-1.5, 1.5; length=81)   # offset added to p_v   (true 1.0)
b_range = range(-1.5, 1.5; length=81)   # offset added to p_v³  (true −1/3)
rows = NamedTuple[]
for κ in (1, 3, 10, 100), a in a_range, b in b_range
    p = copy(FHN_P); p[5] += a; p[10] += b
    push!(rows, (window_size=κ, dp_v=a, dp_v3=b, J=J(p, κ)))
end
write_csv(joinpath(RESULTS, "landscape_2d_v_v3.csv"), rows)

# second plane: (p_w in v̇, p_v in ẇ) — indices 2 and 15 (true −1.0 and 0.08)
rows = NamedTuple[]
for κ in (1, 3, 10, 100), a in a_range, b in range(-0.5, 0.5; length=81)
    p = copy(FHN_P); p[2] += a; p[15] += b
    push!(rows, (window_size=κ, dp_w=a, dp_wv=b, J=J(p, κ)))
end
write_csv(joinpath(RESULTS, "landscape_2d_w_wv.csv"), rows)

# ---- 1-D slices along random unit directions: count local minima ------------------
rng = MersenneTwister(5); srows = NamedTuple[]; crows = NamedTuple[]
ss = range(-2.0, 2.0; length=401)
for dir in 1:12
    u = randn(rng, 20); u ./= norm(u)
    for κ in (1, 2, 5, 10, 25, 50, 100)
        vals = [J(FHN_P .+ s .* u, κ) for s in ss]
        for (s, v) in zip(ss, vals); push!(srows, (direction=dir, window_size=κ, s=s, J=v)); end
        finite = vals .< 1e3
        nmin = count(i -> finite[i] && finite[i-1] && finite[i+1] && vals[i] < vals[i-1] && vals[i] < vals[i+1], 2:length(vals)-1)
        push!(crows, (direction=dir, window_size=κ, n_local_minima=nmin,
                      frac_blowup=1-mean(finite), s_blowup_min=(any(.!finite) ? minimum(abs.(ss[.!finite])) : NaN)))
    end
end
write_csv(joinpath(RESULTS, "landscape_1d_slices.csv"), srows)
write_csv(joinpath(RESULTS, "landscape_1d_minima.csv"), crows)

# ---- Hessian spectrum at p* (data term only) vs κ --------------------------------
hrows = NamedTuple[]
for κ in (1, 2, 3, 5, 10, 20, 25, 50, 100)
    H = ForwardDiff.hessian(p -> J(p, κ; g=0.0), FHN_P)
    H = Symmetric((H .+ H')./2); ev = eigvals(H)
    g = ForwardDiff.gradient(p -> J(p, κ; g=0.0), FHN_P)
    push!(hrows, (window_size=κ, J_true=J(FHN_P, κ; g=0.0), J_true_with_sparsity=J(FHN_P, κ),
                  lambda_min=minimum(ev), lambda_max=maximum(ev), cond=maximum(ev)/max(minimum(ev),1e-300),
                  n_negative=count(<(0), ev), grad_norm=norm(g)))
end
write_csv(joinpath(RESULTS, "hessian_at_ptrue.csv"), hrows)
write_csv(joinpath(RESULTS, "hessian_spectra.csv"),
    [(window_size=κ, index=i, eigenvalue=ev) for κ in (1,5,25,100)
     for (i, ev) in enumerate(eigvals(Symmetric(ForwardDiff.hessian(p -> J(p, κ; g=0.0), FHN_P))))])

# ---- sensitivity of J_κ to the initial-condition variable x0 ------------------------
xrows = NamedTuple[]
for κ in (1, 10, 100), dv in range(-0.5, 0.5; length=41)
    push!(xrows, (window_size=κ, dv0=dv, J=ms_loss(x0 .+ [dv, 0.0], FHN_P, rhs!, data, δt, S, 0.0, κ)))
end
write_csv(joinpath(RESULTS, "landscape_x0_sensitivity.csv"), xrows)
println("03 done")
