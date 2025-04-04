
# Utility function to generate all multi-indices (as vectors of exponents)
# for monomials in d variables of total degree <= deg.
function multiindices(dim::Int, deg::Int)
    if dim == 1
        return [[i] for i in 0:deg]
    else
        indices = Vector{Vector{Int}}()
        for i in 0:deg
            for sub in multiindices(dim-1, deg - i)
                push!(indices, [i; sub])
            end
        end
        return indices
    end
end


function multiindex_mapping(dim::Int, deg::Int; state_names::Union{Nothing, Vector{String}}=nothing)
    # Set default state names if none provided.
    if state_names === nothing
        state_names = ["x_$i" for i in 1:dim]
    elseif length(state_names) != dim
        error("The number of state names must match the dimension.")
    end

    # Internal function to convert a multiindex to its monomial string.
    function _multiindex_to_monomial(index::Vector{Int})
        terms = String[]
        for (exp, name) in zip(index, state_names)
            if exp == 0
                continue
            elseif exp == 1
                push!(terms, name)
            else
                push!(terms, "$name^$exp")
            end
        end
        return isempty(terms) ? "1" : join(terms, " * ")
    end

    # Get all multiindices and build the mapping.
    indices = multiindices(dim, deg)
    mapping = Dict{Tuple{Vararg{Int}}, String}()
    for idx in indices
        mapping[Tuple(idx)] = _multiindex_to_monomial(idx)
    end

    return mapping, [_multiindex_to_monomial(idx) for idx in indices]
end
