# common.jl — shared loader for R001.
# self-contained: frozen deps/ shadows the live repository tree
#
# Every script resolves DEPS FIRST and never reads the live tree. The analysis
# extracts source by LINE RANGE from the frozen files, so a later edit to the
# live sources cannot silently shift what is executed here (see deps/MANIFEST.md).

using SHA

const DEPS = normpath(joinpath(@__DIR__, "..", "deps"))
const RESULTS = joinpath(@__DIR__, "results")
mkpath(RESULTS)

depspath(f) = joinpath(DEPS, f)
depslines(f) = readlines(depspath(f))
depssha(f) = bytes2hex(sha256(read(depspath(f))))

"Execute lines a:b of a FROZEN source file verbatim in Main."
function run_block(f, a, b)
    include_string(Main, join(depslines(f)[a:b], "\n"), "deps/$f:$a")
end

"Assert a frozen file still hashes to what deps/MANIFEST.md recorded."
function check_frozen(f, sha)
    got = depssha(f)
    got == sha || error("deps/$f drifted: expected $sha, got $got")
    return true
end

# --- Line ranges into the frozen sources (deps/MANIFEST.md pins the files) ---
const MOD7 = "fhn_model_selection_mod 7.jl"
const MOD8 = "fhn_model_selection_mod 8.jl"
const MOD6 = "fhn_model_selection_mod 6 stability analysis and visualization.jl"

const MOD7_ODEFUN      = (19, 23)    # function odefun(dy, y, p, t) ... end
const MOD7_DATA        = (60, 107)   # const Np .. cropped+noised `data`; line 108 begins plotting
const MOD7_LOSSES      = (123, 418)  # smoothl1 .. forward_simulation_loss_windows
const MOD7_FSLW        = (293, 418)  # forward_simulation_loss_windows alone
const MOD8_ODEFUN_POLY = (47, 77)    # @inline function odefun_poly! ... end
const MOD8_FSL         = (257, 339)  # function forward_simulation_loss ... end
const MOD6_ODEFUN_NEW  = (980, 993)  # function odefun_new(y, p) ... end

"""
    load_pipeline!()

Load the frozen `integrator.jl` / `helper_functions.jl`, the mod-7 RHS and loss
functions, then build the committed dataset by
executing frozen `mod 7` lines 60–107 verbatim (the plotting block starts at
line 108 and is excluded). Returns a NamedTuple of the derived constants.
"""
function load_pipeline!()
    include(depspath("integrator.jl"))        # frozen Tsit5 integrator
    include(depspath("helper_functions.jl"))  # frozen multi-index helpers
    run_block(MOD7, MOD7_ODEFUN...)
    run_block(MOD7, MOD7_LOSSES...)
    run_block(MOD7, MOD7_DATA...)          # defines Np, fhn_p, fhn_y0, cache, data, ts, Δt, …
    S  = 10
    δt = Main.Δt / S
    return (; S, δt, γ2 = 5e-2,
            data = Main.data, ts = Main.ts, Δt = Main.Δt,
            fhn_p = Main.fhn_p, Nd = div(length(Main.data), 2))
end

"Write a vector of NamedTuples as a CSV (no external dependency)."
function write_csv(path, rows)
    isempty(rows) && (write(path, ""); return path)
    cols = keys(rows[1])
    open(path, "w") do io
        println(io, join(String.(cols), ","))
        for r in rows
            println(io, join((v isa AbstractFloat ? repr(v) : string(v) for v in values(r)), ","))
        end
    end
    return path
end

# --- Minimal JSON writer (JSON.jl is not in this repo's Project.toml, and a
# --- report must not change the environment it is auditing). ---------------
_json(x::AbstractString) = '"' * replace(String(x), '\\'=>"\\\\", '"'=>"\\\"") * '"'
_json(x::Bool)           = x ? "true" : "false"
_json(x::Integer)        = string(x)
_json(x::AbstractFloat)  = isfinite(x) ? repr(x) : "null"
_json(x::AbstractVector) = "[" * join(_json.(x), ", ") * "]"
_json(x::Symbol)         = _json(String(x))
function _json(d::AbstractDict)
    ks = sort(collect(keys(d)), by=string)
    "{" * join(("\n  " * _json(string(k)) * ": " * _json(d[k]) for k in ks), ",") * "\n}"
end
write_json(path, d) = (open(io -> print(io, _json(d)), path, "w"); path)
