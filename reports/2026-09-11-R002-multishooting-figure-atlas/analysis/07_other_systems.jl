# 07 — other systems: does guess propagation (GP) beat restarting from the seed on
# systems OTHER than FitzHugh–Nagumo?  Two systems, using the dimension-agnostic core:
#   lv     : Lotka–Volterra, 2-D, quadratic library (6 monomials × 2 = 12 parameters),
#            NelderMead, arms {propagate, reset} × seeds 1..6
#   lorenz : Lorenz 63, 3-D, quadratic library (10 monomials × 3 = 30 parameters),
#            arms {propagate, reset} × seeds 1..4, run twice: optimizer=:bfgs (AD
#            gradient, 2500-iteration cap) and optimizer=:nm (4000 iterations)
# self-contained: frozen deps/ shadows the live repository tree
#
# Parallelism (as in 04_sweeps.jl): `Distributed` workers cannot open sockets inside the
# sandbox, so the job list is SHARDED over independent processes:
#   julia 07_other_systems.jl <shard> <nshards>   → results/shards/other_*_<shard>.csv
#   julia 07_other_systems.jl merge               → concatenates + derived CSV/JSON
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))
const SHARDS = joinpath(RESULTS, "shards"); mkpath(SHARDS)
const MODE    = length(ARGS) ≥ 1 ? ARGS[1] : "0"
const SHARD   = MODE == "merge" ? 0 : parse(Int, ARGS[1])
const NSHARDS = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 1

const SHORT = [1, 2, 5, 10, 25, 50, 100]

# ------------------------------------------------------------ Lotka–Volterra ----
# library order for PolyLib(2,2): 1, y, y², x, x*y, x²
const LV_LIB = PolyLib(2, 2; names=["x","y"])
# ẋ = αx − βxy ; ẏ = −γy + δxy   with α=1, β=0.5, γ=0.8, δ=0.3  (same as 01_data.jl)
const LV_P = [0.0, 0.0, 0.0, 1.0, -0.5, 0.0,
              0.0, -0.8, 0.0, 0.0,  0.3, 0.0]
lv_dataset() = make_dataset(make_rhs(LV_LIB), LV_P, [2.0, 1.0]; T_end=30.0, δt_fine=0.01,
                            downsample=25, crop=(0.0, 25.0), noise_rel=0.05, seed=2024)

# -------------------------------------------------------------------- Lorenz ----
# library order for PolyLib(3,2) (exponents of (x,y,z) in multiindices(3,2) order):
#   1 | z | z^2 | y | y*z | y^2 | x | x*z | x*y | x^2
const LOR_LIB = PolyLib(3, 2; names=["x","y","z"])
const LOR_SIGMA = 10.0; const LOR_RHO = 28.0; const LOR_BETA = 8/3
# ẋ = σ(y−x)        = −σ x + σ y
# ẏ = x(ρ−z) − y    = ρ x − y − x z
# ż = x y − β z
const LOR_P = [ 0.0,      0.0, 0.0,  LOR_SIGMA, 0.0, 0.0, -LOR_SIGMA,  0.0, 0.0, 0.0,   # dx/dt
                0.0,      0.0, 0.0, -1.0,       0.0, 0.0,  LOR_RHO,   -1.0, 0.0, 0.0,   # dy/dt
                0.0, -LOR_BETA, 0.0,  0.0,      0.0, 0.0,  0.0,        0.0, 1.0, 0.0]   # dz/dt
const LOR_Y0 = [-8.0, 7.0, 27.0]
lorenz_dataset() = make_dataset(make_rhs(LOR_LIB), LOR_P, LOR_Y0; T_end=6.0, δt_fine=0.001,
                                downsample=20, crop=(2.0, 4.0), noise_rel=0.02, seed=77)

"Hand-written Lorenz RHS, independent of the polynomial library."
function lorenz_ref!(du, u, p, t)
    x, y, z = u[1], u[2], u[3]
    du[1] = LOR_SIGMA*(y - x)
    du[2] = x*(LOR_RHO - z) - y
    du[3] = x*y - LOR_BETA*z
    return du
end
"max |make_rhs(LOR_LIB)(u,LOR_P) − lorenz_ref!(u)| over `n` random points in [-30,30]³."
function lorenz_rhs_check(n=64; seed=4242)
    rhs! = make_rhs(LOR_LIB); rng = MersenneTwister(seed)
    worst = 0.0
    for _ in 1:n
        u = 30 .* (2 .* rand(rng, 3) .- 1)
        a = rhs!(zeros(3), u, LOR_P, 0.0); b = lorenz_ref!(zeros(3), u, LOR_P, 0.0)
        worst = max(worst, maximum(abs.(a .- b)))
    end
    return worst
end

# ---------------------------------------------------------------- job driver ----
function job(spec)
    sys = spec[:system]
    if sys == :lv
        lib = LV_LIB; p_true = LV_P; D = lv_dataset()
    else
        lib = LOR_LIB; p_true = LOR_P; D = lorenz_dataset()
    end
    rhs! = make_rhs(lib)
    r = run_sweep(rhs!, D.data, p_true, SHORT; seed=spec[:seed], mode=spec[:mode],
                  optimizer=spec[:optimizer], iters=spec[:iters], γ=5e-2, S=10,
                  x0_scale=spec[:x0_scale])
    meta = (system=String(sys), arm=String(spec[:mode]), seed=spec[:seed],
            optimizer=String(spec[:optimizer]))
    rows  = [merge(meta, row) for row in r.rows]
    mrows = minimizer_rows(r.minimizers; extra=meta)
    return (rows=rows, mrows=mrows, meta=meta, iters=spec[:iters])
end

specs = Dict[]
# A — Lotka–Volterra, NelderMead
for mode in (:propagate, :reset), seed in 1:6
    push!(specs, Dict(:system=>:lv, :mode=>mode, :seed=>seed, :optimizer=>:nm,
                      :iters=>2500, :x0_scale=>0.1, :cost=>1.0))
end
# B — Lorenz, BFGS (AD gradient) and NelderMead
for mode in (:propagate, :reset), seed in 1:4
    push!(specs, Dict(:system=>:lorenz, :mode=>mode, :seed=>seed, :optimizer=>:bfgs,
                      :iters=>2500, :x0_scale=>0.5, :cost=>6.0))
end
for mode in (:propagate, :reset), seed in 1:4
    push!(specs, Dict(:system=>:lorenz, :mode=>mode, :seed=>seed, :optimizer=>:nm,
                      :iters=>4000, :x0_scale=>0.5, :cost=>8.0))
end
# deterministic heavy-first ordering, then round-robin sharding (as in 04_sweeps.jl)
specs = specs[sortperm([-s[:cost] for s in specs])]

if MODE != "merge"
    mine = [specs[i] for i in 1:length(specs) if (i-1) % NSHARDS == SHARD]
    println("shard $SHARD/$NSHARDS: $(length(mine)) of $(length(specs)) jobs"); flush(stdout)
    t0 = time()
    for (k, spec) in enumerate(mine)
        o = job(spec)
        append_csv(joinpath(SHARDS, "other_sweeps_$(SHARD).csv"), o.rows)
        append_csv(joinpath(SHARDS, "other_minimizers_$(SHARD).csv"), o.mrows)
        println("[$SHARD] $k/$(length(mine)) $(o.meta.system)/$(o.meta.arm)/$(o.meta.optimizer)/seed$(o.meta.seed) done, $(round(time()-t0)) s"); flush(stdout)
    end
    write(joinpath(SHARDS, "other_done_$(SHARD)"), "$(length(mine)),$(time()-t0)")
    println("shard $SHARD done in $(round(time()-t0)) s")
    exit(0)
end

# ------------------------------------------------------------------- merge ------
function cat_shards(prefix)
    files = sort(filter(f -> startswith(f, prefix * "_") && endswith(f, ".csv"), readdir(SHARDS)))
    lines = String[]; header = nothing
    for f in files
        ls = readlines(joinpath(SHARDS, f)); isempty(ls) && continue
        header === nothing && (header = ls[1]); ls[1] == header || error("header mismatch in $f")
        append!(lines, ls[2:end])
    end
    header === nothing && return 0
    open(joinpath(RESULTS, prefix * ".csv"), "w") do io
        println(io, header); foreach(l -> println(io, l), lines)
    end
    return length(lines)
end
n_rows  = cat_shards("other_sweeps")
n_mrows = cat_shards("other_minimizers")
done_files = filter(f -> startswith(f, "other_done_"), readdir(SHARDS))
n_done = 0; wall_total = 0.0
for f in done_files
    c = split(read(joinpath(SHARDS, f), String), ",")
    global n_done += parse(Int, c[1]); global wall_total += parse(Float64, c[2])
end
println("merged: $n_rows sweep rows, $n_mrows minimizer rows, from $n_done jobs (of $(length(specs)) planned)")

# ---- datasets and library descriptors -----------------------------------------
DLV  = lv_dataset()
DLOR = lorenz_dataset()
N_lv  = size(DLV.data,1);  Δt_lv  = DLV.data[2,1]  - DLV.data[1,1]
N_lor = size(DLOR.data,1); Δt_lor = DLOR.data[2,1] - DLOR.data[1,1]

write_csv(joinpath(RESULTS, "lorenz_data.csv"),
    [(t=DLOR.data[i,1], x=DLOR.data[i,2], y=DLOR.data[i,3], z=DLOR.data[i,4],
      x_clean=DLOR.clean[i,2], y_clean=DLOR.clean[i,3], z_clean=DLOR.clean[i,4]) for i in 1:N_lor])
write_csv(joinpath(RESULTS, "lorenz_fine.csv"),
    [(t=DLOR.tsall[i], x=DLOR.alldata[i,1], y=DLOR.alldata[i,2], z=DLOR.alldata[i,3])
     for i in 1:5:length(DLOR.tsall)])
write_csv(joinpath(RESULTS, "lorenz_library.csv"),
    [(index=j, monomial=LOR_LIB.labels[(j-1)%10+1],
      equation=(j<=10 ? "dx/dt" : j<=20 ? "dy/dt" : "dz/dt"), p_true=LOR_P[j]) for j in 1:30])

# ---- fitted trajectories from the minimisers (seed 1 of each system/arm/optimizer) ----
mins = Dict{Tuple{String,String,String},Dict{Int,Vector{Float64}}}()
let ls = readlines(joinpath(RESULTS, "other_minimizers.csv"))
    hdr = split(ls[1], ","); ci = Dict(String(h) => i for (i,h) in enumerate(hdr))
    for l in ls[2:end]
        c = split(l, ",")
        parse(Int, c[ci["seed"]]) == 1 || continue
        key = (String(c[ci["system"]]), String(c[ci["arm"]]), String(c[ci["optimizer"]]))
        nz  = key[1] == "lorenz" ? 33 : 14
        κ = parse(Int, c[ci["window_size"]]); j = parse(Int, c[ci["index"]])
        d = get!(mins, key, Dict{Int,Vector{Float64}}())
        get!(d, κ, zeros(nz))[j] = parse(Float64, c[ci["value"]])
    end
end
frows = NamedTuple[]
for key in sort(collect(keys(mins)))
    sys = key[1]
    d      = sys == "lorenz" ? 3 : 2
    rhs!   = make_rhs(sys == "lorenz" ? LOR_LIB : LV_LIB)
    D      = sys == "lorenz" ? DLOR : DLV
    T_end  = D.data[end,1]
    δt     = sys == "lorenz" ? 0.002 : 0.01
    for κ in (1, 10, 100)
        haskey(mins[key], κ) || continue
        z = mins[key][κ]
        ts, X = simulate(rhs!, z[1:d], z[d+1:end], T_end, δt)
        for (i, t) in enumerate(ts)
            any(v -> abs(v) > 1e3 || isnan(v), X[i,:]) && break
            push!(frows, (system=sys, arm=key[2], optimizer=key[3], window_size=κ, t=t,
                          x1=X[i,1], x2=X[i,2], x3=(d == 3 ? X[i,3] : NaN)))
        end
    end
end
write_csv(joinpath(RESULTS, "other_fits.csv"), frows)

# ---- meta ---------------------------------------------------------------------
rhs_check = lorenz_rhs_check()
write_json(joinpath(RESULTS, "other_meta.json"), Dict(
    "n_jobs_planned" => length(specs), "n_jobs_done" => n_done,
    "n_sweep_rows" => n_rows, "n_minimizer_rows" => n_mrows, "n_fit_rows" => length(frows),
    "n_jobs_lv_nm" => count(s -> s[:system]==:lv, specs),
    "n_jobs_lorenz_bfgs" => count(s -> s[:system]==:lorenz && s[:optimizer]==:bfgs, specs),
    "n_jobs_lorenz_nm" => count(s -> s[:system]==:lorenz && s[:optimizer]==:nm, specs),
    "schedule" => SHORT, "gamma" => 5e-2, "S" => 10,
    "lv_N" => N_lv, "lv_Delta_t" => Δt_lv, "lv_T_end" => DLV.data[end,1],
    "lv_noise_rel" => 0.05, "lv_data_seed" => 2024, "lv_n_params" => length(LV_P),
    "lv_x0_scale" => 0.1, "lv_iters" => 2500, "lv_y0" => [2.0, 1.0],
    "lorenz_N" => N_lor, "lorenz_Delta_t" => Δt_lor, "lorenz_T_end" => DLOR.data[end,1],
    "lorenz_noise_rel" => 0.02, "lorenz_data_seed" => 77, "lorenz_n_params" => length(LOR_P),
    "lorenz_x0_scale" => 0.5, "lorenz_iters_bfgs" => 2500, "lorenz_iters_nm" => 4000,
    "lorenz_y0" => LOR_Y0, "lorenz_sigma" => LOR_SIGMA, "lorenz_rho" => LOR_RHO,
    "lorenz_beta" => LOR_BETA, "lorenz_crop" => [2.0, 4.0], "lorenz_delta_t_fine" => 0.001,
    "lorenz_downsample" => 20,
    "lorenz_rhs_max_abs_diff" => rhs_check,
    "lorenz_monomials" => LOR_LIB.labels, "lv_monomials" => LV_LIB.labels,
    "wall_s_total_shards" => wall_total))
println("07 merge done: lv N=$N_lv Δt=$Δt_lv | lorenz N=$N_lor Δt=$Δt_lor | lorenz rhs check=$rhs_check")
