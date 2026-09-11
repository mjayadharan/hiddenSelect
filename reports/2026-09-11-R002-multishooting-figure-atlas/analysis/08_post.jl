# 08 — post-sweep derived quantities (from results/sweep_minimizers.csv, main arms, seed 1):
#   Hessian spectrum of the DATA term at the guess-propagation minimiser p^(κ) for every κ
#   (the strong-convexity premise of the theory, checked where it is actually assumed);
#   the cost along the straight segment between consecutive minimisers (is the carried
#   guess inside the basin of the next problem?); the cost of each minimiser evaluated at
#   every OTHER window size (cross-evaluation matrix).
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))
rhs! = make_rhs(FHN_LIB); D = make_dataset(rhs!, FHN_P, [1.0, 1.0]); data = D.data
Δt = data[2,1]-data[1,1]; S = 10; δt = Δt/S; γ = 5e-2; x0 = data[1,2:3]

lines = readlines(joinpath(RESULTS, "sweep_minimizers.csv")); hdr = split(lines[1], ","); ci = Dict(h=>i for (i,h) in enumerate(hdr))
mins = Dict{Tuple{String,Int},Dict{Int,Vector{Float64}}}()
for l in lines[2:end]
    c = split(l, ","); c[ci["exp"]] == "main" || continue
    key = (String(c[ci["arm"]]), parse(Int, c[ci["seed"]]))
    κ = parse(Int, c[ci["window_size"]]); j = parse(Int, c[ci["index"]]); v = parse(Float64, c[ci["value"]])
    z = get!(get!(mins, key, Dict{Int,Vector{Float64}}()), κ, zeros(22)); z[j] = v
end
Jd(p, κ) = ms_loss(x0, p, rhs!, data, δt, S, 0.0, κ)      # data term only
Jf(z, κ) = ms_loss(view(z,1:2), view(z,3:22), rhs!, data, δt, S, γ, κ)

hrows = NamedTuple[]; prow = NamedTuple[]; xrows = NamedTuple[]
for seed in 1:8
    haskey(mins, ("propagate", seed)) || continue
    M = mins[("propagate", seed)]; ks = sort(collect(keys(M)))
    for κ in ks
        p = M[κ][3:22]
        H = ForwardDiff.hessian(q -> Jd(q, κ), p); H = Symmetric((H .+ H')./2); ev = eigvals(H)
        g = ForwardDiff.gradient(q -> Jd(q, κ), p)
        push!(hrows, (seed=seed, window_size=κ, lambda_min=minimum(ev), lambda_max=maximum(ev), n_negative=count(<(0), ev),
                      grad_norm_data_term=norm(g), J_data=Jd(p, κ), p_err=norm(p-FHN_P)))
    end
    # straight-line path between consecutive minimisers, evaluated with the NEXT window size
    for i in 1:length(ks)-1
        κa, κb = ks[i], ks[i+1]; za, zb = M[κa], M[κb]
        for s in range(0, 1; length=21)
            z = (1-s) .* za .+ s .* zb
            push!(prow, (seed=seed, kappa_from=κa, kappa_to=κb, s=s, J_next=Jf(z, κb), J_prev=Jf(z, κa)))
        end
    end
    # cross-evaluation: minimiser of κ_i evaluated at κ_j
    seed == 2 || continue     # seed 1 starts on the plateau; seed 2 is the representative seed used throughout
    for κi in ks, κj in ks
        push!(xrows, (seed=seed, kappa_min=κi, kappa_eval=κj, J=Jf(M[κi], κj)))
    end
end
write_csv(joinpath(RESULTS, "post_hessian_at_minimizers.csv"), hrows)
write_csv(joinpath(RESULTS, "post_path_between_minimizers.csv"), prow)
write_csv(joinpath(RESULTS, "post_cross_evaluation.csv"), xrows)
println("08 done: $(length(hrows)) hessian rows, $(length(prow)) path rows")
