# 12 — extended cost-landscape grids for the landscape-navigation animations (report addendum).
#
#   Stage `screen` : 16 candidate FHN coefficient planes at 61x61 over the FULL κ ladder, scored
#                    for how DRASTICALLY the landscape changes with κ. Writes the evidence that
#                    justifies the five planes animated below (report §7.1).
#   Stage `grids`  : the five selected FHN planes at 201x201 over the FULL κ ladder — 10.9x the
#                    samples of the 61x61 grid used for the original (v, v^3) animation in
#                    10_animation_data.jl, which is left untouched.
#   Stage `lv`     : the Lotka–Volterra (x^2, xy) plane of ẋ at 201x201 (the 161x161 grid from
#                    11_lv_landscape.jl is left untouched), for the hi-res 2-D and the 3-D animation.
#
# All other coefficients are held at truth; x0 = first datum; γ = 0.05; S = 10 sub-steps.
# Grids are written gzipped (long format, pandas reads .csv.gz directly): a 201x201x16 grid is
# ~18 MB as plain text. gzip is piped through the system binary — a report must not add packages.
#
# Usage: julia --project=<repo> --threads=12 analysis/12_landscape_planes.jl [screen|grids|lv] ...
# self-contained: frozen deps/ shadows the live repository tree
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))

const FULL = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100]
const NSCREEN = 61       # screening grid
const NPROD   = 201      # production grid for the animations

fmt6(x) = string(round(Float64(x), digits=6))
fmt7(x) = (isfinite(x) ? string(round(Float64(x), sigdigits=7)) : "nan")

"Stream a long-format landscape grid straight into a gzip pipe (no new packages)."
function write_grid_gz(path, ca, cb, as, bs, ks, Zof)
    open(path, "w") do out
        gz = open(`gzip -9 -c`, "w", out)
        println(gz, "window_size,$ca,$cb,J")
        for κ in ks
            Z = Zof(κ)
            for ia in eachindex(as), ib in eachindex(bs)
                println(gz, κ, ",", fmt6(as[ia]), ",", fmt6(bs[ib]), ",", fmt7(Z[ia, ib]))
            end
        end
        close(gz)
    end
    return path
end

"J on the (i,j) coefficient plane: offsets `as` x `bs` added to `P0[i]`, `P0[j]`."
function grid(P0, i, j, as, bs, κ, rhs!, data, x0, δt, S, γ)
    Z = Matrix{Float64}(undef, length(as), length(bs))
    Threads.@threads for ia in eachindex(as)
        p = copy(P0)
        for ib in eachindex(bs)
            copyto!(p, P0); p[i] += as[ia]; p[j] += bs[ib]
            Z[ia, ib] = ms_loss(x0, p, rhs!, data, δt, S, γ, κ)
        end
    end
    return Z
end

# ---------------------------------------------------------------- FHN setup ----
fhn_rhs! = make_rhs(FHN_LIB); DF = make_dataset(fhn_rhs!, FHN_P, [1.0, 1.0])
const FDATA = DF.data; const FΔt = FDATA[2,1]-FDATA[1,1]; const FS = 10
const Fδt = FΔt/FS; const Γ = 5e-2; const FX0 = FDATA[1,2:3]

# coefficient labels (LaTeX, consumed by figures/make_landscape_animations.py)
const PLAB = Dict(1=>"constant in \$\\dot v\$", 2=>"coefficient of \$w\$ in \$\\dot v\$",
    4=>"coefficient of \$w^3\$ in \$\\dot v\$", 5=>"coefficient of \$v\$ in \$\\dot v\$",
    6=>"coefficient of \$vw\$ in \$\\dot v\$", 8=>"coefficient of \$v^2\$ in \$\\dot v\$",
    9=>"coefficient of \$v^2w\$ in \$\\dot v\$", 10=>"coefficient of \$v^3\$ in \$\\dot v\$",
    11=>"constant in \$\\dot w\$", 12=>"coefficient of \$w\$ in \$\\dot w\$",
    15=>"coefficient of \$v\$ in \$\\dot w\$", 18=>"coefficient of \$v^2\$ in \$\\dot w\$",
    20=>"coefficient of \$v^3\$ in \$\\dot w\$")

# (key, p-index a, p-index b, range a, range b)   ranges are OFFSETS from truth
const CANDIDATES = [
 ("v_v3",    5, 10, (-1.5, 1.5),   (-1.5, 1.5)),
 ("v_v2",    5,  8, (-1.5, 1.5),   (-0.6, 0.6)),
 ("v3_v2",  10,  8, (-1.0, 1.0),   (-0.6, 0.6)),
 ("w_v",     2,  5, (-1.5, 1.5),   (-1.5, 1.5)),
 ("c_w",     1,  2, (-1.5, 1.5),   (-1.5, 1.5)),
 ("wv_ww",  15, 12, (-0.40, 0.40), (-0.40, 0.40)),
 ("v_wv",    5, 15, (-1.5, 1.5),   (-0.40, 0.40)),
 ("w_wv",    2, 15, (-1.5, 1.5),   (-0.40, 0.40)),
 ("v3_w3",  10, 20, (-1.0, 1.0),   (-0.30, 0.30)),
 ("v3_vw",  10,  6, (-1.0, 1.0),   (-1.0, 1.0)),
 ("c_wc",    1, 11, (-1.5, 1.5),   (-0.30, 0.30)),
 ("v2_w2",   8, 18, (-0.6, 0.6),   (-0.30, 0.30)),
 ("vw_v2w",  6,  9, (-1.0, 1.0),   (-1.0, 1.0)),
 ("v_w3",    5,  4, (-1.5, 1.5),   (-0.6, 0.6)),
 ("w_w3",    2,  4, (-1.5, 1.5),   (-0.6, 0.6)),
 ("v2_v2w",  8,  9, (-0.6, 0.6),   (-1.0, 1.0)),
]
# the five animated planes (chosen from the screen below; see report §7.1)
const SELECTED = ["v_v3", "wv_ww", "v2_w2", "w_wv", "v3_w3"]

blow_frac(Z) = 1 - count(Z .< 1e3)/length(Z)
"Fraction of the plane within a factor 2 of the plane's minimum cost — the visible basin."
basin_frac(Z) = (m = minimum(Z[Z .< 1e3]); count(Z .< 2m)/length(Z))
spearman(u, v) = cor(Float64.(sortperm(sortperm(u))), Float64.(sortperm(sortperm(v))))

stages = isempty(ARGS) ? ["screen", "grids", "lv"] : ARGS

# ------------------------------------------------------------------ screen ----
if "screen" in stages
    rows = NamedTuple[]; summ = NamedTuple[]
    for (key, i, j, ra, rb) in CANDIDATES
        as = collect(range(ra...; length=NSCREEN)); bs = collect(range(rb...; length=NSCREEN))
        Zs = Dict(κ => grid(FHN_P, i, j, as, bs, κ, fhn_rhs!, FDATA, FX0, Fδt, FS, Γ) for κ in FULL)
        for κ in FULL
            Z = Zs[κ]; fin = Z .< 1e3
            push!(rows, (plane=key, window_size=κ, f_blow=blow_frac(Z), basin=basin_frac(Z),
                         J_min=minimum(Z[fin]), J_max=maximum(Z[fin])))
        end
        Z1, Z2 = Zs[1], Zs[100]; both = (Z1 .< 1e3) .& (Z2 .< 1e3)
        ρ = spearman(log.(Z1[both]), log.(Z2[both]))
        b1, b2 = basin_frac(Z1), basin_frac(Z2)
        dr(Z) = log10(maximum(Z[Z .< 1e3]) / minimum(Z[Z .< 1e3]))
        push!(summ, (plane=key, p_index_a=i, p_index_b=j,
                     rank_decorrelation=1-ρ, basin_k1=b1, basin_k100=b2,
                     basin_shrink=b1/max(b2, 1/NSCREEN^2),
                     f_blow_k1=blow_frac(Z1), f_blow_k100=blow_frac(Z2),
                     dyn_range_decades_k1=dr(Z1), dyn_range_decades_k100=dr(Z2),
                     selected=(key in SELECTED)))
        println(rpad(key, 9), " 1-ρ=", rpad(round(1-ρ, digits=2), 5),
                " basin ", rpad(round(b1, digits=4), 6), "->", rpad(round(b2, digits=5), 7),
                " (x", rpad(round(b1/max(b2, 1/NSCREEN^2), digits=1), 6), ")",
                " blow ", rpad(round(blow_frac(Z1), digits=2), 4), "->", rpad(round(blow_frac(Z2), digits=2), 4),
                " decades ", rpad(round(dr(Z1), digits=2), 5), "->", round(dr(Z2), digits=2)); flush(stdout)
    end
    write_csv(joinpath(RESULTS, "landscape_plane_screen.csv"), rows)
    write_csv(joinpath(RESULTS, "landscape_plane_screen_summary.csv"), summ)
    println("screen done: ", length(CANDIDATES), " planes x ", length(FULL), " κ at $(NSCREEN)x$(NSCREEN)")
end

# ------------------------------------------------------------- FHN grids ------
if "grids" in stages
    meta = Dict{String,Any}()
    for (key, i, j, ra, rb) in CANDIDATES
        key in SELECTED || continue
        as = collect(range(ra...; length=NPROD)); bs = collect(range(rb...; length=NPROD))
        ca, cb = "d$(key)_a", "d$(key)_b"
        t0 = time()
        write_grid_gz(joinpath(RESULTS, "anim_fhn_landscape_$(key).csv.gz"), ca, cb, as, bs, FULL,
                      κ -> grid(FHN_P, i, j, as, bs, κ, fhn_rhs!, FDATA, FX0, Fδt, FS, Γ))
        meta[key] = Dict("system"=>"fhn", "p_index_a"=>i, "p_index_b"=>j,
                         "z_index_a"=>i+2, "z_index_b"=>j+2,      # z = [x0 (2 entries); p]
                         "truth_a"=>FHN_P[i], "truth_b"=>FHN_P[j],
                         "col_a"=>ca, "col_b"=>cb, "label_a"=>PLAB[i], "label_b"=>PLAB[j],
                         "n"=>NPROD, "seed"=>2, "file"=>"anim_fhn_landscape_$(key).csv.gz")
        println("grid $key: $(NPROD)x$(NPROD)x$(length(FULL)) in ", round(time()-t0, digits=1), " s"); flush(stdout)
    end
    write_json(joinpath(RESULTS, "landscape_planes_meta.json"), meta)
end

# ------------------------------------------------- Lotka–Volterra hi-res ------
if "lv" in stages
    LV = PolyLib(2, 2; names=["x","y"]); LV_P = [0.0,0.0,0.0,1.0,-0.5,0.0, 0.0,-0.8,0.0,0.0,0.3,0.0]
    lv_rhs! = make_rhs(LV)
    DL = make_dataset(lv_rhs!, LV_P, [2.0, 1.0]; T_end=30.0, δt_fine=0.01, downsample=25,
                      crop=(0.0, 25.0), noise_rel=0.05, seed=2024)
    ld = DL.data; lΔt = ld[2,1]-ld[1,1]; lS = 10; lδt = lΔt/lS; lx0 = ld[1,2:3]
    i, j = 6, 5                                   # x^2 and xy in ẋ (LV library order: 1,y,y²,x,xy,x²)
    as = collect(range(-0.6, 0.6; length=NPROD)); bs = collect(range(-1.5, 0.8; length=NPROD))
    t0 = time()
    write_grid_gz(joinpath(RESULTS, "anim_lv_landscape_x2_xy_hires.csv.gz"), "dp_x2", "dp_xy", as, bs, FULL,
                  κ -> grid(LV_P, i, j, as, bs, κ, lv_rhs!, ld, lx0, lδt, lS, Γ))
    println("LV x2_xy hi-res: $(NPROD)x$(NPROD)x$(length(FULL)) in ", round(time()-t0, digits=1), " s")
    write_json(joinpath(RESULTS, "landscape_lv_hires_meta.json"),
        Dict("x2_xy"=>Dict("system"=>"lv", "p_index_a"=>i, "p_index_b"=>j, "z_index_a"=>i+2, "z_index_b"=>j+2,
             "truth_a"=>LV_P[i], "truth_b"=>LV_P[j], "col_a"=>"dp_x2", "col_b"=>"dp_xy",
             "label_a"=>"coefficient of \$x^2\$ in \$\\dot x\$", "label_b"=>"coefficient of \$xy\$ in \$\\dot x\$",
             "n"=>NPROD, "seed"=>1, "file"=>"anim_lv_landscape_x2_xy_hires.csv.gz")))
end
println("12 done")
