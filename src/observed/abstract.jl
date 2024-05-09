"""
    samples(observed::SemObservedData)

Gets the matrix of observed data samples.
Rows are samples, columns are observed variables.

## See Also
[`nsamples`](@ref), [`observed_vars`](@ref).
"""
samples(observed::SemObserved) = observed.data
nsamples(observed::SemObserved) = observed.nsamples

observed_vars(observed::SemObserved) = observed.observed_vars

############################################################################################
### Additional functions
############################################################################################

# compute the permutation that subsets and reorders source elements
# to match the destination order.
# if multiple identical elements are present in the source, the last one is used.
# if one_to_one is true, checks that the source and destination have the same length.
function source_to_dest_perm(
    src::AbstractVector,
    dest::AbstractVector;
    one_to_one::Bool = false,
    entities::String = "elements",
)
    if dest == src # exact match
        return eachindex(dest)
    else
        one_to_one &&
            length(dest) != length(src) &&
            throw(
                DimensionMismatch(
                    "The length of the new $entities order ($(length(dest))) " *
                    "does not match the number of $entities ($(length(src)))",
                ),
            )
        src_inds = Dict(el => i for (i, el) in enumerate(src))
        return [src_inds[el] for el in dest]
    end
end
