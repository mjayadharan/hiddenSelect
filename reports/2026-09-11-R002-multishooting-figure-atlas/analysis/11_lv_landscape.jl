# 11 — Lotka–Volterra 2-D cost landscape in the (x, xy) coefficient plane of the ẋ equation
#      (library indices 4 and 5; true values 1.0 and −0.5), at every FULL window size, for
#      the LV landscape animation. Other coefficients at truth, x0 = first datum, γ = 0.05.
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))
FULL = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100]
LV = PolyLib(2, 2; names=["x","y"]); LV_P = [0.0,0.0,0.0,1.0,-0.5,0.0, 0.0,-0.8,0.0,0.0,0.3,0.0]
rhs! = make_rhs(LV)
D = make_dataset(rhs!, LV_P, [2.0, 1.0]; T_end=30.0, δt_fine=0.01, downsample=25, crop=(0.0, 25.0), noise_rel=0.05, seed=2024)
data = D.data; Δt = data[2,1]-data[1,1]; S = 10; δt = Δt/S; γ = 5e-2; x0 = data[1,2:3]
rows = NamedTuple[]
ar = range(-1.5, 1.5; length=61); br = range(-1.0, 1.0; length=61)
for κ in FULL, a in ar, b in br
    p = copy(LV_P); p[4] += a; p[5] += b
    push!(rows, (window_size=κ, dp_x=a, dp_xy=b, J=ms_loss(x0, p, rhs!, data, δt, S, γ, κ)))
end
write_csv(joinpath(RESULTS, "anim_lv_landscape_x_xy.csv"), rows)
println("11 done: ", length(rows), " rows")
