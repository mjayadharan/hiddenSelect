# 02 — numerics behind the concept and theory figures:
#   single vs multiple shooting at a WRONG parameter; Lemma 1 (flow sensitivity to x0);
#   Lemma 2 (flow sensitivity to p); Proposition 1 (node removal); Lipschitz constants.
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))

rhs! = make_rhs(FHN_LIB)
D = make_dataset(rhs!, FHN_P, [1.0, 1.0])
data = D.data; N = size(data,1); Δt = data[2,1]-data[1,1]; S = 10; δt = Δt/S
T_end = data[end,1]

# ---- (a) shooting segments at a wrong parameter --------------------------------
# p_wrong: cubic coefficient of v̇ scaled to −0.30 (true −1/3) and ẇ time-scale ×1.15
p_wrong = copy(FHN_P); p_wrong[10] = -0.30; p_wrong[11:20] .*= 1.15
segrows = NamedTuple[]
for (label, p) in (("true", FHN_P), ("wrong", p_wrong)), κ in (1, 5, 10, 25, N-1)
    cache = Tsit5Cache(zeros(2)); x = cache.ycur
    win = 0
    for i in 1:N-1
        if (i-1) % κ == 0
            win += 1; x .= data[i, 2:3]
            push!(segrows, (param=label, window_size=κ, window=win, t=data[i,1], v=x[1], w=x[2], is_node=true))
        end
        for s in 1:S
            integration_step!(cache, rhs!, x, 0.0, p, δt, false)
            push!(segrows, (param=label, window_size=κ, window=win, t=data[i,1]+s*δt, v=x[1], w=x[2], is_node=false))
        end
    end
end
write_csv(joinpath(RESULTS, "concept_segments.csv"), segrows)
# costs of the two parameter vectors at each κ (no sparsity term)
write_csv(joinpath(RESULTS, "concept_costs.csv"),
    [(param=l, window_size=κ, J=ms_loss(data[1,2:3], p, rhs!, data, δt, S, 0.0, κ))
     for (l,p) in (("true",FHN_P),("wrong",p_wrong)), κ in (1,2,5,10,25,50,N-1)])

# ---- (b) Lipschitz constants along the true trajectory ----------------------------
# L(p*) = sup_t ||∂f/∂x||₂ along the attractor, L̃ = sup_t ||∂f/∂p||₂
jac_x(u, p) = ForwardDiff.jacobian(u_ -> rhs!(similar(u_), u_, p, 0.0), u)
jac_p(u, p) = ForwardDiff.jacobian(p_ -> rhs!(similar(p_, 2), u, p_, 0.0), p)
Lx = maximum(opnorm(jac_x(D.alldata[i,:], FHN_P)) for i in 1:size(D.alldata,1))
Lp = maximum(opnorm(jac_p(D.alldata[i,:], FHN_P)) for i in 1:size(D.alldata,1))
Lx_box = maximum(opnorm(jac_x([v,w], FHN_P)) for v in range(-2.5,2.5,length=41), w in range(-1,2,length=41))
# log-norm (one-sided Lipschitz) along the trajectory: sup λ_max( (J+Jᵀ)/2 )
μlog = maximum(maximum(eigvals(Symmetric((J .+ J')./2))) for J in (jac_x(D.alldata[i,:], FHN_P) for i in 1:size(D.alldata,1)))

# ---- (c) Lemma 1: ‖φ(t;x1)−φ(t;x2)‖/‖x1−x2‖ vs t -------------------------------
rng = MersenneTwister(3); lrows = NamedTuple[]
ts_probe = collect(0.0:0.1:10.0)
for start in (1, 21, 41, 61), k in 1:6
    x1 = data[start, 2:3]; dir = randn(rng, 2); dir ./= norm(dir); ε = 1e-3
    x2 = x1 .+ ε .* dir
    _, X1 = simulate(rhs!, x1, FHN_P, 10.0, 0.01); _, X2 = simulate(rhs!, x2, FHN_P, 10.0, 0.01)
    for (j, t) in enumerate(ts_probe)
        i = Int(round(t/0.01)) + 1
        push!(lrows, (start_index=start, direction=k, t=t, ratio=norm(X1[i,:]-X2[i,:])/ε))
    end
end
write_csv(joinpath(RESULTS, "lemma1_flow_sensitivity.csv"), lrows)

# ---- (d) Lemma 2: ‖φ(t;p1)−φ(t;p2)‖/‖p1−p2‖ vs t ----------------------------------
prow = NamedTuple[]
for start in (1, 21, 41, 61), k in 1:6
    x0 = data[start, 2:3]; dp = randn(rng, 20); dp ./= norm(dp); ε = 1e-3
    _, X1 = simulate(rhs!, x0, FHN_P, 10.0, 0.01); _, X2 = simulate(rhs!, x0, FHN_P .+ ε .* dp, 10.0, 0.01)
    for t in ts_probe
        i = Int(round(t/0.01)) + 1
        push!(prow, (start_index=start, direction=k, t=t, ratio=norm(X1[i,:]-X2[i,:])/ε))
    end
end
write_csv(joinpath(RESULTS, "lemma2_param_sensitivity.csv"), prow)

# ---- (e) Proposition 1: node removal changes the cost by a bounded amount ----------
# Fine partition: every datum is a node (κ=1). Remove |I_R| nodes; measure |Ĵ − J|.
"Cost with an explicit node set (indices into the data rows), no sparsity term."
function cost_with_nodes(p, nodes::Vector{Int})
    cache = Tsit5Cache(zeros(2)); x = cache.ycur; loss = 0.0
    nodeset = Set(nodes)
    x .= data[1,2:3]
    for i in 1:N-1
        (i in nodeset) && (x .= data[i,2:3])
        for _ in 1:S; integration_step!(cache, rhs!, x, 0.0, p, δt, false); end
        loss += sum(abs2, x .- data[i+1,2:3])
    end
    return loss/N
end
nrows = NamedTuple[]
for (label, p) in (("true", FHN_P), ("near", FHN_P .+ 0.02 .* [1,1,0,0,1,0,0,0,0,1, 1,1,0,0,1,0,0,0,0,0]),
                   ("wrong", p_wrong))
    J_fine = cost_with_nodes(p, collect(1:N-1))
    # remove every m-th node (|I_R| grows, ΔT₂ = Δt fixed) — "remove few at a time"
    for stride in (2, 3, 4, 5, 10, 20, 50)
        nodes = [i for i in 1:N-1 if (i-1) % stride == 0]
        push!(nrows, (param=label, experiment="stride", stride=stride, n_removed=(N-1)-length(nodes),
                      DeltaT2=Δt, DeltaT1=stride*Δt, J_fine=J_fine, J_coarse=cost_with_nodes(p, nodes),
                      absdiff=abs(cost_with_nodes(p, nodes)-J_fine)))
    end
    # remove one contiguous block of b nodes after node 30 (|I_R| = b, ΔT₂ = b·Δt grows)
    for b in (1, 2, 3, 5, 8, 12, 20)
        nodes = [i for i in 1:N-1 if !(31 ≤ i ≤ 30+b)]
        Jc = cost_with_nodes(p, nodes)
        push!(nrows, (param=label, experiment="block", stride=0, n_removed=b, DeltaT2=b*Δt, DeltaT1=(b+1)*Δt,
                      J_fine=J_fine, J_coarse=Jc, absdiff=abs(Jc-J_fine)))
    end
end
write_csv(joinpath(RESULTS, "prop1_node_removal.csv"), nrows)

write_json(joinpath(RESULTS, "concept_meta.json"), Dict(
    "L_traj" => Lx, "L_box" => Lx_box, "Ltilde_traj" => Lp, "lognorm_traj" => μlog,
    "p_wrong_v3" => p_wrong[10], "p_wrong_w_scale" => 1.15, "p_wrong_err" => norm(p_wrong-FHN_P),
    "epsilon" => 1e-3, "probe_T" => 10.0))
println("02 done: L=$Lx Lbox=$Lx_box Ltilde=$Lp lognorm=$μlog")
