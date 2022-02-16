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
        data = nothing,
        obs_cov = nothing,
        meanstructure = false,
        rowwise = false,
        kwargs...)
    # if no cov. matrix was given, compute one
    if isnothing(obs_cov) obs_cov = Statistics.cov(data) end
    isnothing(data) ? n_obs = nothing : n_obs = convert(Float64, size(data, 1))
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