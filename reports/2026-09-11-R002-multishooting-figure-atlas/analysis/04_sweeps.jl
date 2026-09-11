# 04 — the parallel experiment batch (Distributed, one process per job):
#   main      : guess propagation vs best-guess vs reset (control), 8 seeds, full schedule
#   schedule  : how the window-size schedule (few vs many nodes removed per step) matters
#   penalty   : graded blow-up penalty vs the repository's flat plateau
#   noise     : GP vs reset across observation-noise levels
#   sparsity  : sparsity weight γ sweep (GP)
#   optimizer : NelderMead vs BFGS vs LBFGS (AD gradient through the windowed loss; R001-F4 fixed)
#   basin     : basin of attraction of a single window size from p* + r·u
#   iters     : NelderMead iteration budget
#   substeps  : integrator sub-steps S per data interval
# self-contained: frozen deps/ shadows the live repository tree
#
# Parallelism: `Distributed` workers cannot open sockets inside the sandbox, so the
# job list is SHARDED over independent processes:  julia 04_sweeps.jl <shard> <nshards>
# writes results/shards/*_<shard>.csv ; `julia 04_sweeps.jl merge` concatenates them.
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))
const SHARDS = joinpath(RESULTS, "shards"); mkpath(SHARDS)
const MODE = length(ARGS) ≥ 1 ? ARGS[1] : "0"
const SHARD = MODE == "merge" ? 0 : parse(Int, ARGS[1])
const NSHARDS = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 1
const TAG = get(ENV, "R002_TAG", "")               # distinguishes a second batch's shard files
const ONLY = split(get(ENV, "R002_ONLY_EXPS", ""), ",")   # optional: restrict to these exps

const FULL  = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100]
const SHORT = [1, 2, 5, 10, 25, 50, 100]
const DENSE = vcat(collect(1:30), collect(35:5:100))

function job(spec)
    rhs! = make_rhs(FHN_LIB)
    D = make_dataset(rhs!, FHN_P, [1.0, 1.0]; noise_rel=get(spec, :noise, 0.05))
    r = run_sweep(rhs!, D.data, FHN_P, spec[:schedule]; seed=spec[:seed], mode=spec[:mode],
                  optimizer=get(spec, :optimizer, :nm), iters=get(spec, :iters, 2500),
                  γ=get(spec, :gamma, 5e-2), S=get(spec, :S, 10), penalty=get(spec, :penalty, :flat),
                  x0_scale=get(spec, :x0_scale, 0.1), trace_every=get(spec, :trace, 0),
                  p_init=get(spec, :p_init, nothing))
    meta = (exp=String(spec[:exp]), arm=String(spec[:arm]), seed=spec[:seed], mode=String(spec[:mode]),
            optimizer=String(get(spec, :optimizer, :nm)), iters=get(spec, :iters, 2500),
            gamma=get(spec, :gamma, 5e-2), S=get(spec, :S, 10), penalty=String(get(spec, :penalty, :flat)),
            noise=get(spec, :noise, 0.05), schedule=String(get(spec, :schedule_name, "full")),
            radius=get(spec, :radius, NaN), n_steps=length(spec[:schedule]))
    rows = [merge(meta, row) for row in r.rows]
    mrows = minimizer_rows(r.minimizers; extra=(exp=meta.exp, arm=meta.arm, seed=meta.seed))
    trows = NamedTuple[]
    for (κ, tr) in r.traces, (it, v) in enumerate(tr)
        push!(trows, (exp=meta.exp, arm=meta.arm, seed=meta.seed, window_size=κ, iteration=it-1, J=v))
    end
    return (rows=rows, mrows=mrows, trows=trows, minimizers=r.minimizers, meta=meta)
end

specs = Dict[]
# main arms
for mode in (:propagate, :best, :reset), seed in 1:8
    push!(specs, Dict(:exp=>:main, :arm=>mode, :mode=>mode, :seed=>seed, :schedule=>FULL, :trace=>(seed==1 ? 1 : 0)))
end
# schedules (propagate)
for (name, sch) in (("dense", DENSE), ("coarse", [1, 5, 25, 100]), ("jump", [1, 100])), seed in 1:8
    push!(specs, Dict(:exp=>:schedule, :arm=>name, :mode=>:propagate, :seed=>seed, :schedule=>sch, :schedule_name=>name))
end
# graded penalty
for mode in (:propagate, :reset), seed in 1:8
    push!(specs, Dict(:exp=>:penalty, :arm=>Symbol(String(mode)*"_graded"), :mode=>mode, :seed=>seed, :schedule=>FULL, :penalty=>:graded))
end
# noise
for σ in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2), mode in (:propagate, :reset), seed in 1:6
    push!(specs, Dict(:exp=>:noise, :arm=>mode, :mode=>mode, :seed=>seed, :schedule=>SHORT, :noise=>σ, :schedule_name=>"short"))
end
# sparsity weight
for γ in (0.0, 1e-3, 1e-2, 5e-2, 0.2, 1.0), seed in 1:6   # arm carries γ so the minimisers are distinguishable
    push!(specs, Dict(:exp=>:sparsity, :arm=>Symbol("gamma$(γ)"), :mode=>:propagate, :seed=>seed, :schedule=>SHORT, :gamma=>γ, :schedule_name=>"short"))
end
# optimizers (propagate and reset)
for opt in (:bfgs, :lbfgs), mode in (:propagate, :reset), seed in 1:8
    push!(specs, Dict(:exp=>:optimizer, :arm=>Symbol(String(opt)*"_"*String(mode)), :mode=>mode, :seed=>seed, :schedule=>FULL, :optimizer=>opt))
end
# basin of attraction: one window size, start at p* + r·u
let rng = MersenneTwister(99)
    for κ in (1, 5, 100), r in (0.05, 0.1, 0.2, 0.4, 0.8), seed in 1:10
        u = randn(rng, 20); u ./= norm(u)
        push!(specs, Dict(:exp=>:basin, :arm=>Symbol("k$(κ)"), :mode=>:reset, :seed=>seed, :schedule=>[κ], :radius=>r,
                          :p_init=>FHN_P .+ r .* u, :schedule_name=>"single"))
    end
end
# iteration budget
for it in (250, 500, 1000, 2500, 5000), seed in 1:4
    push!(specs, Dict(:exp=>:iters, :arm=>Symbol("it$(it)"), :mode=>:propagate, :seed=>seed, :schedule=>SHORT, :iters=>it, :schedule_name=>"short"))
end
# integrator sub-steps
for S in (1, 2, 5, 10, 20), seed in 1:4
    push!(specs, Dict(:exp=>:substeps, :arm=>Symbol("S$(S)"), :mode=>:propagate, :seed=>seed, :schedule=>SHORT, :S=>S, :schedule_name=>"short"))
end
# seed scale: how far from p* the random start is (0.01 = repository mod 7/8, 0.1 = R001) — flat vs graded penalty
for sc in (0.01, 0.03, 0.1, 0.3), mode in (:propagate, :reset), pen in (:flat, :graded), seed in 1:8
    push!(specs, Dict(:exp=>:seedscale, :arm=>Symbol("sc$(sc)_$(mode)_$(pen)"), :mode=>mode, :seed=>seed, :schedule=>SHORT,
                      :x0_scale=>sc, :penalty=>pen, :schedule_name=>"short"))
end
# deterministic heavy-first ordering, then round-robin sharding
order = sortperm([ -length(s[:schedule]) * (get(s,:iters,2500)/2500) for s in specs])
specs = specs[order]
if MODE != "merge"
    mine = [specs[i] for i in 1:length(specs) if (i-1) % NSHARDS == SHARD]
    ONLY == [""] || (mine = [sp for sp in mine if String(sp[:exp]) in ONLY])
    # resume: skip (exp, arm, seed) already present in this shard's file
    donekeys = Set{String}()
    for f in filter(x -> startswith(x, "sweeps_") && endswith(x, ".csv"), readdir(SHARDS))   # ALL shards: assignment may change
        ls = readlines(joinpath(SHARDS, f)); isempty(ls) && continue
        h = split(ls[1], ","); idx = [findfirst(==(k), h) for k in ("exp", "arm", "seed", "noise", "radius", "gamma", "iters", "S")]
        for l in ls[2:end]; c = split(l, ","); push!(donekeys, join((c[i] for i in idx), "/")); end
    end
    jobkey(sp) = join((String(sp[:exp]), String(sp[:arm]), string(sp[:seed]), _fmt(get(sp, :noise, 0.05)), _fmt(get(sp, :radius, NaN)),
                       _fmt(get(sp, :gamma, 5e-2)), string(get(sp, :iters, 2500)), string(get(sp, :S, 10))), "/")
    mine = [sp for sp in mine if !(jobkey(sp) in donekeys)]
    println("shard $SHARD/$NSHARDS: $(length(mine)) of $(length(specs)) jobs (after resume skip)"); flush(stdout)
    t0 = time()
    for (k, spec) in enumerate(mine)
        o = job(spec)
        append_csv(joinpath(SHARDS, "sweeps_$(TAG)$(SHARD).csv"), o.rows)
        append_csv(joinpath(SHARDS, "sweep_minimizers_$(TAG)$(SHARD).csv"), o.mrows)
        isempty(o.trows) || append_csv(joinpath(SHARDS, "sweep_traces_$(TAG)$(SHARD).csv"), o.trows)
        println("[$SHARD] $k/$(length(mine)) $(o.meta.exp)/$(o.meta.arm)/seed$(o.meta.seed) done, $(round(time()-t0)) s"); flush(stdout)
    end
    write(joinpath(SHARDS, "done_$(TAG)$(SHARD)"), string(length(mine)))
    println("shard $SHARD done in $(round(time()-t0)) s")
    exit(0)
end

# ---------------- merge ----------------
function cat_shards(prefix)
    files = sort(filter(f -> startswith(f, prefix * "_") && endswith(f, ".csv"), readdir(SHARDS)))
    lines = String[]; header = nothing
    for f in files
        ls = readlines(joinpath(SHARDS, f)); isempty(ls) && continue
        header === nothing && (header = ls[1]); ls[1] == header || error("header mismatch in $f")
        append!(lines, ls[2:end])
    end
    header === nothing && return 0
    let h = split(header, ",")     # drop exact-duplicate cells (deterministic reruns across batches); key excludes timing columns
        keycols = [i for (i, k) in enumerate(h) if !(k in ("wall_s", "f_calls", "iterations", "converged"))]
        seen = Set{String}(); keep = String[]
        for l in lines
            c = split(l, ","); k = join((c[i] for i in keycols), ",")
            k in seen || (push!(seen, k); push!(keep, l))
        end
        lines = keep
    end
    let h = split(header, ","); ie = findfirst(==("exp"), h); ia = findfirst(==("arm"), h)
        if ie !== nothing && ia !== nothing   # superseded first-batch sparsity rows (arm renamed to gammaX)
            filter!(l -> (c = split(l, ","); !(c[ie] == "sparsity" && c[ia] == "propagate")), lines)
        end
    end
    open(joinpath(RESULTS, prefix * ".csv"), "w") do io
        println(io, header); foreach(l -> println(io, l), lines)
    end
    return length(lines)
end
n_rows = cat_shards("sweeps"); cat_shards("sweep_minimizers"); cat_shards("sweep_traces")
n_done = sum(parse(Int, read(joinpath(SHARDS, f), String)) for f in readdir(SHARDS) if startswith(f, "done_"))
println("merged: $n_rows sweep rows from $n_done jobs (of $(length(specs)) planned)")
# re-run seed-1 main arms cheaply? No — rebuild minimizers from the merged file instead.
mins_rows = readlines(joinpath(RESULTS, "sweep_minimizers.csv"))
hdr = split(mins_rows[1], ","); ci = Dict(h => i for (i,h) in enumerate(hdr))
ok = []
let store = Dict{Tuple{String,String,Int},Dict{Int,Vector{Float64}}}()
    for l in mins_rows[2:end]
        c = split(l, ","); key = (String(c[ci["exp"]]), String(c[ci["arm"]]), parse(Int, c[ci["seed"]]))
        key[1] == "main" && key[3] in (1, 2) && key[2] in ("propagate","reset") || continue
        κ = parse(Int, c[ci["window_size"]]); j = parse(Int, c[ci["index"]]); v = parse(Float64, c[ci["value"]])
        d = get!(store, key, Dict{Int,Vector{Float64}}()); z = get!(d, κ, zeros(22)); z[j] = v
    end
    for (key, mins) in store
        push!(ok, (meta=(exp=key[1], arm=key[2], seed=key[3]), minimizers=mins))
    end
end

# fitted trajectories along the sweep (seed 1, main propagate + reset) for the filmstrip/animation
rhs! = make_rhs(FHN_LIB); D = make_dataset(rhs!, FHN_P, [1.0, 1.0]); T_end = D.data[end,1]
frows = NamedTuple[]
for o in ok
    (o.meta.exp == "main" && o.meta.seed in (1, 2) && o.meta.arm in ("propagate", "reset")) || continue
    for (κ, z) in o.minimizers
        p = z[3:end]; x0 = z[1:2]
        ts, X = simulate(rhs!, x0, p, T_end, 0.05)
        for (i, t) in enumerate(ts)
            (abs(X[i,1]) > 1e3 || isnan(X[i,1])) && break
            push!(frows, (arm=o.meta.arm, seed=o.meta.seed, window_size=κ, t=t, v=X[i,1], w=X[i,2]))
        end
    end
end
write_csv(joinpath(RESULTS, "sweep_fits.csv"), frows)
write_json(joinpath(RESULTS, "sweeps_meta.json"), Dict(
    "n_jobs_planned" => length(specs), "n_jobs_done" => n_done, "n_sweep_rows" => n_rows, "schedule_full" => FULL, "schedule_short" => SHORT, "schedule_dense" => DENSE,
    "nm_iterations_default" => 2500, "x0_scale" => 0.1, "blowup_penalty" => 1000.0))
println("04 merge done")
