struct SemObsCommon{
        A <: Union{AbstractArray, Nothing},
        B <: AbstractArray,
        C <: Union{AbstractArray, Nothing},
        D <: AbstractFloat,
        O <: Union{AbstractFloat, Nothing}} <: SemObs
    data::A
    obs_cov::B
    obs_mean::C
    n_man::D
    n_obs::O
end

function SemObsCommon(;
        data = nothing,
        obs_cov = nothing,
        meanstructure = false)
    # if no cov. matrix was given, compute one
    if isnothing(obs_cov) obs_cov = Statistics.cov(data) end
    isnothing(data) ? n_obs = nothing : n_obs = convert(Float64, size(data, 1))
    n_man = Float64(size(obs_cov, 1))
    # if a meanstructure is needed, compute observed means
    meanstructure ? obs_mean = vcat(Statistics.mean(data, dims = 1)...) :
        obs_mean = nothing
    return SemObsCommon(data, obs_cov, obs_mean, n_man, n_obs)
end

struct SemObsMissing{
        A <: AbstractArray,
        D <: AbstractFloat,
        O <: AbstractFloat,
        P <: Vector,
        R <: Vector,
        PD <: AbstractArray,
        PO <: AbstractArray,
        PVO <: AbstractArray} <: SemObs
    data::A
    n_man::D
    n_obs::O
    patterns::P # missing patterns
    rows::R # coresponding rows in data_rowwise
    data_rowwise::PD # list of data
    pattern_n_obs::PO # observed rows per pattern
    pattern_nvar_obs::PVO # number of non-missing variables per pattern
end

function SemObsMissing(data)

    # remove persons with only missings
    keep = Vector{Int64}()
    for i = 1:size(data, 1)
        if any(.!ismissing.(data[i, :]))
            push!(keep, i)
        end
    end
    data = data[keep, :]

    n_obs = size(data, 1)
    n_man = size(data, 2)
    

    # compute and store the different missing patterns with their rowindices
    missings = ismissing.(data)
    patterns = [missings[i, :] for i = 1:size(missings, 1)]

    patterns_cart = findall.(!, patterns)
    data_rowwise = [data[i, patterns_cart[i]] for i = 1:n_obs]
    data_rowwise = convert.(Array{Float64}, data_rowwise)

    remember = Vector{BitArray{1}}()
    rows = [Vector{Int64}(undef, 0) for i = 1:size(patterns, 1)]
    for i = 1:size(patterns, 1)
        unknown = true
        for j = 1:size(remember, 1)
            if patterns[i] == remember[j]
                push!(rows[j], i)
                unknown = false
            end
        end
        if unknown
            push!(remember, patterns[i])
            push!(rows[size(remember, 1)], i)
        end
    end
    rows = rows[1:length(remember)]
    n_patterns = size(rows, 1)

    # sort by number of missings
    sort_n_miss = sortperm(sum.(remember))
    remember = remember[sort_n_miss]
    remember_cart = findall.(!, remember)
    rows = rows[sort_n_miss]

    pattern_n_obs = size.(rows, 1)
    pattern_nvar_obs = length.(remember_cart) 

    return SemObsMissing(data, Float64(n_man), Float64(n_obs), remember_cart,
    rows, data_rowwise, Float64.(pattern_n_obs), Float64.(pattern_nvar_obs))
end
