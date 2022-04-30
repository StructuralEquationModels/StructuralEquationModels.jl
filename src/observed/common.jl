"""
    More here soon.
"""
struct SemObsCommon{
        A <: Union{AbstractArray, Nothing},
        B <: AbstractArray,
        C <: Union{AbstractArray, Nothing},
        D <: AbstractFloat,
        O <: Union{AbstractFloat, Nothing},
        R <: Union{AbstractArray, Nothing}} <: SemObs
    data::A
    obs_cov::B
    obs_mean::C
    n_man::D
    n_obs::O
    data_rowwise::R
end

function SemObsCommon(;
        specification = nothing,
        data = nothing,
        spec_colnames = nothing,
        obs_cov = nothing,
        data_colnames = nothing,
        meanstructure = false,
        rowwise = false,
        n_obs = nothing,
        kwargs...)

    # sort columns/rows
    if isnothing(spec_colnames) spec_colnames = get_colnames(specification) end

    data, obs_cov = reorder_observed(data, obs_cov, spec_colnames, data_colnames)

    # if no cov. matrix was given, compute one
    if isnothing(obs_cov) obs_cov = Statistics.cov(data) end
    isnothing(data) ? nothing : n_obs = convert(Float64, size(data, 1))
    n_man = Float64(size(obs_cov, 1))
    # if a meanstructure is needed, compute observed means
    meanstructure ? 
        obs_mean = vcat(Statistics.mean(data, dims = 1)...) :
        obs_mean = nothing
    rowwise ? 
        data_rowwise = [data[i, :] for i = 1:convert(Int64, n_obs)] :
        data_rowwise = nothing
    return SemObsCommon(data, obs_cov, obs_mean, n_man, n_obs, data_rowwise)
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemObsCommon)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end

############################################################################
### Recommended methods
############################################################################

n_obs(observed::SemObsCommon) = observed.n_obs
n_man(observed::SemObsCommon) = observed.n_man

############################################################################
### additional methods
############################################################################

get_data(observed::SemObsCommon) = observed.data
obs_cov(observed::SemObsCommon) = observed.obs_cov
obs_mean(observed::SemObsCommon) = observed.obs_mean
data_rowwise(observed::SemObsCommon) = observed.data_rowwise

############################################################################
### Additional functions
############################################################################

# specification colnames
function get_colnames(specification::ParameterTable)
    if !haskey(specification.variables, :sorted_vars) || (length(specification.variables[:sorted_vars]) == 0)
        colnames = specification.variables[:observed_vars]
    else
        is_obs = [var ∈ specification.variables[:observed_vars] for var in specification.variables[:sorted_vars]]
        colnames = specification.variables[:sorted_vars][is_obs]
    end
    return colnames
end

function get_colnames(specification::RAMMatrices)
    if isnothing(specification.colnames)
        @warn "Your RAMMatrices do not contain column names. Please make shure the order of variables in your data is correct!"
        return nothing
    else
        colnames = specification.colnames[specification.F_ind]
        return colnames
    end
end

function get_colnames(specification::Nothing)
    return nothing
end

# reorder data to match spec_colnames --------------------------------------------------------------
reorder_observed(data::Nothing, obs_cov, spec_colnames, data_colnames) = reorder_obs_cov(obs_cov, spec_colnames, data_colnames)
reorder_observed(data, obs_cov::Nothing, spec_colnames, data_colnames) = reorder_data(data, spec_colnames, data_colnames)
reorder_observed(data::Nothing, obs_cov, spec_colnames::Nothing, data_colnames) = nothing, obs_cov
reorder_observed(data, obs_cov::Nothing, spec_colnames::Nothing, data_colnames) = data, nothing

# too much or not enough data specified
reorder_observed(data, obs_cov, spec_colnames, data_colnames) = 
    throw(ArgumentError("you specified both an observed dataset and an observed covariance matrix"))
reorder_observed(data::Nothing, obs_cov::Nothing, spec_colnames, data_colnames) = 
    throw(ArgumentError("you specified neither an observed dataset nor an observed covariance matrix"))

# reorder data ------------------------------------------------------------------------------------------------------------
reorder_data(data::AbstractArray, spec_colnames, data_colnames::Nothing) =
    throw(ArgumentError("if your data format does not provide column names, please provide them via the `data_colnames = ...` argument."))

function reorder_data(data::AbstractArray, spec_colnames, data_colnames)
    if spec_colnames == data_colnames
        return data, nothing
    else
        new_position = [findall(x .== data_colnames)[1] for x in spec_colnames]
        data = Matrix(data[:, new_position])
        return data, nothing
    end
end

reorder_data(data::DataFrame, spec_colnames, data_colnames::Nothing) = Matrix(data[:, spec_colnames]), nothing
reorder_data(data::DataFrame, spec_colnames, data_colnames) = 
    throw(ArgumentError("your data format has column names but you also provided column names via the `data_colnames = ...` argument."))

# reorder covariance matrices ---------------------------------------------------------------------------------------------
reorder_obs_cov(obs_cov::AbstractArray, spec_colnames, data_colnames::Nothing) =
    throw(ArgumentError("If an observed covariance is given, `data_colnames = ...` has to be specified."))

function reorder_obs_cov(obs_cov::AbstractArray, spec_colnames, data_colnames)
    new_position = [findall(x .== data_colnames)[1] for x in spec_colnames]
    indices = reshape([CartesianIndex(i, j) for j in new_position for i in new_position], size(obs_cov, 1), size(obs_cov, 1))
    obs_cov = obs_cov[indices]
    return nothing, obs_cov
end

