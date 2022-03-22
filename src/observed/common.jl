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
        colnames = nothing,
        obs_cov = nothing,
        cov_colnames = nothing,
        meanstructure = false,
        rowwise = false,
        n_obs = nothing,
        kwargs...)

    # sort columns/rows
    if isnothing(colnames) colnames = get_colnames(specification) end
    
    if !isnothing(data) 
        if !isnothing(colnames)
            data = data[:, colnames]
        end
        data = Matrix(data)
    end

    if !isnothing(obs_cov) && isnothing(cov_colnames)
        @error "An observed covariance was given, but no cov_colnames where specified"
    end

    if !isnothing(obs_cov) && !isnothing(cov_colnames)
        new_position = [findall(x .== cov_colnames)[1] for x in colnames]
        indices = reshape([CartesianIndex(i, j) for j in new_position for i in new_position], size(obs_cov, 1), size(obs_cov, 1))
        obs_cov = obs_cov[indices]
    end

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

############################################################################
### Additional functions
############################################################################

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