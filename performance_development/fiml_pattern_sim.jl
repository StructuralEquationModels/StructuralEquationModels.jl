using Random, Distributions, Plots

function cum_miss(; nobs, nvar, p)
    b = Bernoulli(p)
    sample = rand(b, nobs*nvar)
    sample = reshape(sample, (nobs, nvar))
    patterns = [sample[i, :] for i in 1:size(sample, 1)]

    #binomial = sum.(patterns)
    #histogram(binomial)

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
    n_pattern = size(rows, 1)

    # sort by number of missings
    sort_n_miss = sortperm(sum.(remember))
    remember = remember[sort_n_miss]
    remember_cart = findall.(!, remember)
    rows = rows[sort_n_miss]

    pattern_n_obs = size.(rows, 1)
    pattern_nvar_obs = length.(remember_cart)

    obs_pat = pattern_n_obs.*pattern_nvar_obs
    sort_n_miss = sortperm(obs_pat, rev = true)
    obs_pat = obs_pat[sort_n_miss]
    pattern_n_obs = pattern_n_obs[sort_n_miss]

    #pattern_n_obs_cum = pattern_n_obs_cum[sort_n_miss]
    pattern_nvar_obs = pattern_nvar_obs[sort_n_miss]
    #complexity = complexity[sort_n_miss]

    pattern_n_obs_cum = cumsum(pattern_n_obs)
    
    complexity = pattern_nvar_obs.^3
    obs_pat_cum = cumsum(obs_pat)

    

    
    p1 = plot(1:n_pattern, obs_pat_cum, title = "observed datapoints cum.", ylims = (0, Inf))
    p2 = plot(1:n_pattern, pattern_n_obs, title = "N observed")
    p3 = plot(1:n_pattern, pattern_n_obs_cum, title = "N observed cum.", ylims = (0, Inf))
    p4 = plot(1:n_pattern, pattern_nvar_obs, title = "N variables")
    p5 = plot(1:n_pattern, complexity, title = "N variables ^3")

    plot(p1, p2, p3, p4, layout = (2,2))
end

cum_miss(;nobs= 1000, nvar= 10, p= 0.1)

cum_miss(;nobs= 1000, nvar= 100, p= 0.1)

cum_miss(;nobs= 5000, nvar= 100, p= 0.1)

cum_miss(;nobs= 1000, nvar= 100, p= 0.01)

cum_miss(;nobs= 10000, nvar= 100, p= 0.01)

cum_miss(;nobs= 1000, nvar= 10, p= 0.9)


a = [Base.binomial(100, k) for k = 0:5]

function sim_dropout(nwaves, dist)
    pattern = falses(nwaves)
    sample = rand(dist, nwaves)
    t_dropout = findfirst(sample)
    !isnothing(t_dropout) ?
        pattern[t_dropout:nwaves] .= true :
        nothing
    pattern[1] = false
    return pattern
end

function get_miss(nvar, dist, miss_wave)
    if miss_wave
        pattern = trues(nvar)
    else
        pattern = rand(dist, nvar)
    end
    return pattern
end

function pattern_long(nvar, dist, miss_waves)
    pattern = 
        [get_miss(nvar, dist, miss_wave) for miss_wave in miss_waves]
    pattern = vcat(pattern...)
    return(pattern)
end

function cum_miss_dropout(; nobs, nvar, nwaves, p, p_dropout)
    dist1 = Bernoulli(p_dropout)
    dist2 = Bernoulli(p)
    miss_waves = [sim_dropout(nwaves, dist1) for i in 1:nobs]
        
    patterns = [pattern_long(nvar, dist2, miss_waves[i]) for i in 1:nobs]

    #return patterns
    #binomial = sum.(patterns)
    #histogram(binomial)

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
    n_pattern = size(rows, 1)

    # sort by number of missings
    sort_n_miss = sortperm(sum.(remember))
    remember = remember[sort_n_miss]
    remember_cart = findall.(!, remember)
    rows = rows[sort_n_miss]

    pattern_n_obs = size.(rows, 1)
    pattern_nvar_obs = length.(remember_cart)

    obs_pat = pattern_n_obs.*pattern_nvar_obs
    sort_n_miss = sortperm(obs_pat, rev = true)
    obs_pat = obs_pat[sort_n_miss]
    pattern_n_obs = pattern_n_obs[sort_n_miss]

    #pattern_n_obs_cum = pattern_n_obs_cum[sort_n_miss]
    pattern_nvar_obs = pattern_nvar_obs[sort_n_miss]
    #complexity = complexity[sort_n_miss]

    pattern_n_obs_cum = cumsum(pattern_n_obs)
    
    complexity = pattern_nvar_obs.^3
    obs_pat_cum = cumsum(obs_pat)

    

    
    p1 = plot(1:n_pattern, obs_pat_cum, title = "observed datapoints cum.", ylims = (0, Inf))
    p2 = plot(1:n_pattern, pattern_n_obs, title = "N observed")
    p3 = plot(1:n_pattern, pattern_n_obs_cum, title = "N observed cum.", ylims = (0, Inf))
    p4 = plot(1:n_pattern, pattern_nvar_obs, title = "N variables")
    p5 = plot(1:n_pattern, complexity, title = "N variables ^3")

    plot(p1, p2, p3, p4, layout = (2,2))
end

cum_miss_dropout(nobs = 1000, nvar = 10, nwaves = 10, p = 0.05, p_dropout = 0.1)


dist = Binomial(10, 0.1)

