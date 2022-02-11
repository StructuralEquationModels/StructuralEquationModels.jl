struct SemObsMissing{
        A <: AbstractArray,
        D <: AbstractFloat,
        O <: AbstractFloat,
        P <: Vector,
        P2 <: Vector,
        R <: Vector,
        PD <: AbstractArray,
        PO <: AbstractArray,
        PVO <: AbstractArray,
        A2 <: AbstractArray,
        A3 <: AbstractArray
        } <: SemObs
    data::A
    n_man::D
    n_obs::O
    patterns::P # missing patterns
    patterns_not::P2
    rows::R # coresponding rows in data_rowwise
    data_rowwise::PD # list of data
    pattern_n_obs::PO # observed rows per pattern
    pattern_nvar_obs::PVO # number of non-missing variables per pattern
    obs_mean::A2
    obs_cov::A3
end

function SemObsMissing(;data, kwargs...)

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
    remember_cart_not = findall.(remember)
    rows = rows[sort_n_miss]

    pattern_n_obs = size.(rows, 1)
    pattern_nvar_obs = length.(remember_cart) 

    cov_mean = [cov_and_mean(data_rowwise[rows]) for rows in rows]
    obs_cov = [cov_mean[1] for cov_mean in cov_mean]
    obs_mean = [cov_mean[2] for cov_mean in cov_mean]

    return SemObsMissing(data, Float64(n_man), Float64(n_obs), remember_cart,
    remember_cart_not, 
    rows, data_rowwise, Float64.(pattern_n_obs), Float64.(pattern_nvar_obs),
    obs_mean, obs_cov)
end