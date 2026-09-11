# 11 — Lotka–Volterra 2-D cost landscapes at every FULL window size (for the landscape
#      animations). Other coefficients at truth, x0 = first datum, γ = 0.05. Planes:
#        x_xy   : (x, xy) coefficients of ẋ            (true 1.0, −0.5)   61×61
#        x2_xy  : (x², xy) coefficients of ẋ           (true 0.0, −0.5)  161×161 — a true-zero
#                 quadratic term: positive values blow up in finite time (cf. the FHN cubic)
#        xy_xy  : (xy in ẋ, xy in ẏ), the interaction pair (true −0.5, 0.3)  161×161
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))
FULL = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100]
LV = PolyLib(2, 2; names=["x","y"]); LV_P = [0.0,0.0,0.0,1.0,-0.5,0.0, 0.0,-0.8,0.0,0.0,0.3,0.0]
rhs! = make_rhs(LV)
D = make_dataset(rhs!, LV_P, [2.0, 1.0]; T_end=30.0, δt_fine=0.01, downsample=25, crop=(0.0, 25.0), noise_rel=0.05, seed=2024)
data = D.data; Δt = data[2,1]-data[1,1]; S = 10; δt = Δt/S; γ = 5e-2; x0 = data[1,2:3]
planes = [("x_xy", 4, 5, (-1.5, 1.5), (-1.0, 1.0), 61, "dp_x", "dp_xy"),
          ("x2_xy", 6, 5, (-0.6, 0.6), (-1.5, 0.8), 161, "dp_x2", "dp_xy"),
          ("xy_xy", 5, 11, (-1.0, 1.0), (-0.8, 0.8), 161, "dp_xy_x", "dp_xy_y")]
which = length(ARGS) ≥ 1 ? split(ARGS[1], ",") : [p[1] for p in planes]
for (name, i, j, ra, rb, n, ca, cb) in planes
    name in which || continue
    rows = NamedTuple[]
    for κ in FULL, a in range(ra...; length=n), b in range(rb...; length=n)
        p = copy(LV_P); p[i] += a; p[j] += b
        push!(rows, NamedTuple{(:window_size, Symbol(ca), Symbol(cb), :J)}((κ, a, b, ms_loss(x0, p, rhs!, data, δt, S, γ, κ))))
    end
    write_csv(joinpath(RESULTS, "anim_lv_landscape_$(name).csv"), rows)
    println("plane $name: ", length(rows), " rows"); flush(stdout)
end
println("11 done")
