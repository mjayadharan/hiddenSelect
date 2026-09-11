# common.jl — shared loader for R002 (figure atlas).
# self-contained: frozen deps/ shadows the live repository tree
#
# Every script resolves DEPS first and never reads the live tree. `deps/` holds
# the POST-FIX sources (the state committed with this report); `deps/prefix/`
# holds the PRE-FIX snapshot at commit 44abf4a, used only by the bug scripts.
using SHA

const DEPS    = normpath(joinpath(@__DIR__, "..", "deps"))
const PREFIX  = joinpath(DEPS, "prefix")
const RESULTS = joinpath(@__DIR__, "results")
mkpath(RESULTS)
const REPO_PROJECT = normpath(joinpath(@__DIR__, "..", "..", ".."))

depspath(f)   = joinpath(DEPS, f)
prefixpath(f) = joinpath(PREFIX, f)
depssha(f)    = bytes2hex(sha256(read(depspath(f))))
prefixsha(f)  = bytes2hex(sha256(read(prefixpath(f))))

"Execute lines a:b of a frozen source file verbatim in module `m` (default Main)."
function run_block(path, a, b; m=Main)
    include_string(m, join(readlines(path)[a:b], "\n"), "$(basename(path)):$a")
end

const MOD8 = "fhn_model_selection_mod 8.jl"
const MOD7 = "fhn_model_selection_mod 7.jl"
const MOD6 = "fhn_model_selection_mod 6 stability analysis and visualization.jl"

# line ranges (identical in deps/ and deps/prefix/ for mod 8: the fixes replaced lines 1:1)
const MOD8_ODEFUN      = (19, 23)
const MOD8_ODEFUN_POLY = (47, 77)
const MOD8_PARAMS      = (82, 92)     # fhn_p, const Np, fhn_y0
const MOD8_DATA        = (95, 157)    # multi_index_set .. Δt   (needs DifferentialEquations)
const MOD8_SMOOTHL1    = (178, 185)
const MOD8_FSL         = (278, 353)   # forward_simulation_loss
const MOD8_FSLW        = (422, 562)   # forward_simulation_loss_windows
const MOD7_FSLW_PREFIX = (293, 418)
const MOD7_FSLW_FIXED  = (293, 420)
const MOD6_ODEFUN_NEW_PREFIX = (980, 993)
const MOD6_ODEFUN_NEW_FIXED  = (980, 996)

# ---------- minimal CSV / JSON writers (no new packages: a report must not
# ---------- modify the environment it audits) ----------------------------------
_fmt(v::AbstractFloat) = isfinite(v) ? repr(v) : "nan"
_fmt(v) = string(v)
function write_csv(path, rows)
    isempty(rows) && (write(path, ""); return path)
    cols = keys(rows[1])
    open(path, "w") do io
        println(io, join(String.(cols), ","))
        for r in rows
            println(io, join((_fmt(v) for v in values(r)), ","))
        end
    end
    return path
end
function append_csv(path, rows)  # header only if the file is new
    isempty(rows) && return path
    new = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        new && println(io, join(String.(keys(rows[1])), ","))
        for r in rows
            println(io, join((_fmt(v) for v in values(r)), ","))
        end
    end
    return path
end
"Long-format matrix writer: one row per (row index, column name, value)."
function write_matrix_csv(path, names, M::AbstractMatrix)
    open(path, "w") do io
        println(io, join(names, ","))
        for i in 1:size(M,1)
            println(io, join((_fmt(M[i,j]) for j in 1:size(M,2)), ","))
        end
    end
    return path
end
_json(x::AbstractString) = '"' * replace(String(x), '\\'=>"\\\\", '"'=>"\\\"") * '"'
_json(x::Bool)           = x ? "true" : "false"
_json(x::Integer)        = string(x)
_json(x::AbstractFloat)  = isfinite(x) ? repr(x) : "null"
_json(x::Nothing)        = "null"
_json(x::AbstractVector) = "[" * join(_json.(x), ", ") * "]"
_json(x::Symbol)         = _json(String(x))
function _json(d::AbstractDict)
    ks = sort(collect(keys(d)), by=string)
    "{" * join(("\n  " * _json(string(k)) * ": " * _json(d[k]) for k in ks), ",") * "\n}"
end
write_json(path, d) = (open(io -> print(io, _json(d)), path, "w"); path)
