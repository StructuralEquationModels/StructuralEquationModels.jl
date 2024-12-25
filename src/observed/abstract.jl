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

# generate default observed variable names if none provided
default_observed_vars(nobserved_vars::Integer, prefix::Union{Symbol, AbstractString}) =
    Symbol.(prefix, 1:nobserved_vars)

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

# function to prepare input data shared by SemObserved implementations
# returns tuple of
# 1) the matrix of data
# 2) the observed variable symbols that match matrix columns
# 3) the permutation of the original observed_vars (nothing if no reordering)
# If observed_vars is not specified, the vars order is taken from the specification.
# If both observed_vars and specification are provided, the observed_vars are used to match
# the column of the user-provided data matrix, and observed_vars(specification) is used to
# reorder the columns of the data to match the speciation.
# If no variable names are provided at all, generates the symbols in the form
# Symbol(observed_var_prefix, i) for i=1:nobserved_vars.
function prepare_data(
    data::Union{AbstractDataFrame, AbstractMatrix, Nothing},
    observed_vars::Union{AbstractVector, Nothing},
    spec::Union{SemSpecification, Nothing},
    nobserved_vars::Union{Integer, Nothing} = nothing;
    observed_var_prefix::Union{Symbol, AbstractString},
)
    obs_vars = nothing
    obs_vars_perm = nothing
    if !isnothing(observed_vars)
        obs_vars = Symbol.(observed_vars)
        if !isnothing(spec)
            obs_vars_spec = SEM.observed_vars(spec)
            try
                obs_vars_perm = source_to_dest_perm(
                    obs_vars,
                    obs_vars_spec,
                    one_to_one = false,
                    entities = "observed_vars",
                )
            catch err
                if isa(err, KeyError)
                    throw(
                        ArgumentError(
                            "observed_var \"$(err.key)\" from SEM specification is not listed in observed_vars argument",
                        ),
                    )
                else
                    rethrow(err)
                end
            end
            # ignore trivial reorder
            if obs_vars_perm == eachindex(obs_vars)
                obs_vars_perm = nothing
            end
        end
    elseif !isnothing(spec)
        obs_vars = SEM.observed_vars(spec)
    end
    # observed vars in the order that matches the specification
    obs_vars_reordered = isnothing(obs_vars_perm) ? obs_vars : obs_vars[obs_vars_perm]

    # subset the data, check that obs_vars matches data or guess the obs_vars
    if data isa AbstractDataFrame
        if !isnothing(obs_vars_reordered) # subset/reorder columns
            data = data[:, obs_vars_reordered]
        else # default symbol names
            obs_vars = obs_vars_reordered = Symbol.(names(data))
        end
        data_mtx = Matrix(data)
    elseif data isa AbstractMatrix
        if !isnothing(obs_vars)
            size(data, 2) == length(obs_vars) || DimensionMismatch(
                "The number of columns in the data matrix ($(size(data, 2))) does not match the length of observed_vars ($(length(obs_vars))).",
            )
            # reorder columns to match the spec
            data_ordered = !isnothing(obs_vars_perm) ? data[:, obs_vars_perm] : data
        else
            obs_vars =
                obs_vars_reordered =
                    default_observed_vars(size(data, 2), observed_var_prefix)
            data_ordered = data
        end
        # make sure data_mtx is a dense matrix (required for methods like mean_and_cov())
        data_mtx =
            data_ordered isa DenseMatrix ? data_ordered : convert(Matrix, data_ordered)
    elseif isnothing(data)
        data_mtx = nothing
        if !isnothing(nobserved_vars)
            if isnothing(obs_vars)
                obs_vars =
                    obs_vars_reordered =
                        default_observed_vars(nobserved_vars, observed_var_prefix)
            end
        else
            error("Cannot infer observed variables from provided inputs.")
        end
    else
        throw(ArgumentError("Unsupported data type: $(typeof(data))"))
    end
    # check if obs_vars matches nobserved_vars
    # note that obs_vars_reordered may be shorter due to spec-based subsetting
    if !isnothing(obs_vars) &&
       !isnothing(nobserved_vars) &&
       length(obs_vars) != nobserved_vars
        DimensionMismatch(
            "The length of observed_vars ($(length(obs_vars))) does not match nobserved_vars=$(nobserved_vars).",
        )
    end
    return data_mtx, obs_vars_reordered, obs_vars_perm
end
