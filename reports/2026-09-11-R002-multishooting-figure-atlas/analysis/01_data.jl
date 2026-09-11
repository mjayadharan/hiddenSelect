# 01 — datasets and static descriptors (FHN + Lotka–Volterra), for the concept figures.
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))

rhs! = make_rhs(FHN_LIB)
D = make_dataset(rhs!, FHN_P, [1.0, 1.0])
N = size(D.data,1); Δt = D.data[2,1]-D.data[1,1]

# noisy data, clean samples, fine trajectory
write_csv(joinpath(RESULTS, "fhn_data.csv"),
    [(t=D.data[i,1], v=D.data[i,2], w=D.data[i,3], v_clean=D.clean[i,2], w_clean=D.clean[i,3]) for i in 1:N])
write_csv(joinpath(RESULTS, "fhn_fine.csv"),
    [(t=D.tsall[i], v=D.alldata[i,1], w=D.alldata[i,2]) for i in 1:5:length(D.tsall)])

# phase-plane vector field + nullclines on a grid (for the "problem" figure)
vs = range(-2.5, 2.5; length=25); ws = range(-1.0, 2.0; length=25)
rows = NamedTuple[]
for v in vs, w in ws
    du = rhs!(zeros(2), [v,w], FHN_P, 0.0)
    push!(rows, (v=v, w=w, dv=du[1], dw=du[2]))
end
write_csv(joinpath(RESULTS, "fhn_vector_field.csv"), rows)

# library + true coefficient matrix (for the "sparse selection" figure)
write_csv(joinpath(RESULTS, "fhn_library.csv"),
    [(index=j, monomial=FHN_LIB.labels[(j-1)%10+1], equation=(j<=10 ? "dv/dt" : "dw/dt"), p_true=FHN_P[j])
     for j in 1:20])

# noise statistics used by the theory bounds
ηmax = maximum(sqrt.((D.data[:,2]-D.clean[:,2]).^2 + (D.data[:,3]-D.clean[:,3]).^2))
write_json(joinpath(RESULTS, "fhn_data_meta.json"), Dict(
    "N" => N, "Delta_t" => Δt, "noise_rel" => 0.05, "data_seed" => 1287436679,
    "sigma_abs_v" => 0.05*std(D.clean[:,2]), "sigma_abs_w" => 0.05*std(D.clean[:,3]),
    "eta_max_norm" => ηmax, "eta_rms" => sqrt(mean((D.data[:,2:3]-D.clean[:,2:3]).^2)),
    "v_range" => collect(extrema(D.clean[:,2])), "w_range" => collect(extrema(D.clean[:,3])),
    "noise_floor_J" => mean(sum((D.data[:,2:3]-D.clean[:,2:3]).^2, dims=2))))

# ---- Lotka–Volterra (quadratic library, 12 parameters) -------------------------
LV = PolyLib(2, 2; names=["x","y"])            # 1, y, y², x, xy, x²
# ẋ = x(α − βy) = αx − βxy ; ẏ = −y(γ − δx) = −γy + δxy ; α=1, β=0.5, γ=0.8, δ=0.3
LV_P = [0.0, 0.0, 0.0, 1.0, -0.5, 0.0,
        0.0, -0.8, 0.0, 0.0, 0.3, 0.0]
lv_rhs! = make_rhs(LV)
DL = make_dataset(lv_rhs!, LV_P, [2.0, 1.0]; T_end=30.0, δt_fine=0.01, downsample=25, crop=(0.0, 25.0),
                  noise_rel=0.05, seed=2024)
write_csv(joinpath(RESULTS, "lv_data.csv"),
    [(t=DL.data[i,1], x=DL.data[i,2], y=DL.data[i,3], x_clean=DL.clean[i,2], y_clean=DL.clean[i,3]) for i in 1:size(DL.data,1)])
write_csv(joinpath(RESULTS, "lv_fine.csv"),
    [(t=DL.tsall[i], x=DL.alldata[i,1], y=DL.alldata[i,2]) for i in 1:5:length(DL.tsall)])
write_csv(joinpath(RESULTS, "lv_library.csv"),
    [(index=j, monomial=LV.labels[(j-1)%6+1], equation=(j<=6 ? "dx/dt" : "dy/dt"), p_true=LV_P[j]) for j in 1:12])
println("01 done: N=$N Δt=$Δt ηmax=$ηmax  LV N=$(size(DL.data,1))")
