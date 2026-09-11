# 10 — extra data for the animations (report addendum): 2-D landscape slices at every FULL
#      window size (landscape morphing), concept segments at more window sizes (node removal),
#      Lotka–Volterra fits at every stage of the sweep (seed 1, GP vs control), and the FHN
#      GP/control coefficient path (seed 2) projected on the landscape plane.
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))
FULL = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100]

rhs! = make_rhs(FHN_LIB); D = make_dataset(rhs!, FHN_P, [1.0, 1.0]); data = D.data
N = size(data,1); Δt = data[2,1]-data[1,1]; S = 10; δt = Δt/S; γ = 5e-2; x0 = data[1,2:3]

# (a) landscape morphing: (p_v, p_v3) plane, 61×61, every FULL κ
rows = NamedTuple[]
ar = range(-1.5, 1.5; length=61); br = range(-1.5, 1.5; length=61)
for κ in FULL, a in ar, b in br
    p = copy(FHN_P); p[5] += a; p[10] += b
    push!(rows, (window_size=κ, dp_v=a, dp_v3=b, J=ms_loss(x0, p, rhs!, data, δt, S, γ, κ)))
end
write_csv(joinpath(RESULTS, "anim_landscape_v_v3.csv"), rows)

# (b) concept segments (wrong p) at every FULL κ
p_wrong = copy(FHN_P); p_wrong[10] = -0.30; p_wrong[11:20] .*= 1.15
seg = NamedTuple[]
for κ in FULL
    cache = Tsit5Cache(zeros(2)); x = cache.ycur; win = 0
    for i in 1:N-1
        if (i-1) % κ == 0
            win += 1; x .= data[i,2:3]
            push!(seg, (window_size=κ, window=win, t=data[i,1], v=x[1], w=x[2], is_node=true))
        end
        for s in 1:S
            integration_step!(cache, rhs!, x, 0.0, p_wrong, δt, false)
            push!(seg, (window_size=κ, window=win, t=data[i,1]+s*δt, v=x[1], w=x[2], is_node=false))
        end
    end
end
write_csv(joinpath(RESULTS, "anim_concept_segments.csv"), seg)
write_csv(joinpath(RESULTS, "anim_concept_costs.csv"),
    [(window_size=κ, J_wrong=ms_loss(x0, p_wrong, rhs!, data, δt, S, 0.0, κ), J_true=ms_loss(x0, FHN_P, rhs!, data, δt, S, 0.0, κ)) for κ in FULL])

# (c) Lotka–Volterra fits at every stage, seed 1, both arms (from other_minimizers.csv)
LV = PolyLib(2, 2; names=["x","y"]); LV_P = [0.0,0.0,0.0,1.0,-0.5,0.0, 0.0,-0.8,0.0,0.0,0.3,0.0]
lv_rhs! = make_rhs(LV)
DL = make_dataset(lv_rhs!, LV_P, [2.0, 1.0]; T_end=30.0, δt_fine=0.01, downsample=25, crop=(0.0, 25.0), noise_rel=0.05, seed=2024)
lines = readlines(joinpath(RESULTS, "other_minimizers.csv")); hdr = split(lines[1], ","); ci = Dict(h=>i for (i,h) in enumerate(hdr))
mins = Dict{Tuple{String,Int},Dict{Int,Vector{Float64}}}()
for l in lines[2:end]
    c = split(l, ","); c[ci["system"]] == "lv" || continue
    key = (String(c[ci["arm"]]), parse(Int, c[ci["seed"]])); κ = parse(Int, c[ci["window_size"]])
    z = get!(get!(mins, key, Dict{Int,Vector{Float64}}()), κ, zeros(14)); z[parse(Int, c[ci["index"]])] = parse(Float64, c[ci["value"]])
end
frows = NamedTuple[]
for seed in 1:2, arm in ("propagate", "reset")
    haskey(mins, (arm, seed)) || continue
    for (κ, z) in mins[(arm, seed)]
        ts, X = simulate(lv_rhs!, z[1:2], z[3:end], DL.data[end,1], 0.02)
        for (i, t) in enumerate(ts)
            (abs(X[i,1]) > 1e3 || isnan(X[i,1])) && break
            push!(frows, (arm=arm, seed=seed, window_size=κ, t=t, x=X[i,1], y=X[i,2]))
        end
    end
end
write_csv(joinpath(RESULTS, "anim_lv_fits.csv"), frows)
println("10 done: ", length(rows), " landscape rows, ", length(seg), " segment rows, ", length(frows), " LV fit rows")
