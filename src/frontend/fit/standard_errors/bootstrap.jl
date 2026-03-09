"""
    se_bootstrap(
        sem_fit::SemFit;
        n_boot = 3000,
        data = nothing,
        specification = nothing,
        parallel = false,
        kwargs...)

Return bootstrap standard errors.

# Arguments
- `n_boot`: number of boostrap samples
- `data`: data to sample from. Only needed if different than the data from `sem_fit`
- `specification`: a `ParameterTable` or `RAMMatrices` object passed down to `replace_observed`.
   Necessary for FIML models.
- `parallel`: if `true`, run bootstrap samples in parallel on all available threads.
  The number of threads is controlled by the `JULIA_NUM_THREADS` environment variable or
  the `--threads` flag when starting Julia.
- `kwargs...`: passed down to `replace_observed`
"""
function se_bootstrap(
    fitted::SemFit{Mi, So, St, Mo, O};
    n_boot = 3000,
    data = nothing,
    specification = nothing,
    parallel = false,
    kwargs...,
) where {Mi, So, St, Mo, O}
    # access data and convert to matrix
    data = prepare_data_bootstrap(data, fitted.model)
    start = solution(fitted)
    # pre-allocations
    total_sum         = zero(start)
    total_squared_sum = zero(start)
    n_failed          = Ref(0)
    # fit to bootstrap samples
    if !parallel
        for _ in 1:n_boot
            sample_data = bootstrap_sample(data)
            new_model = replace_observed(
                fitted.model;
                data = sample_data, specification = specification, kwargs...)
            try
                sol = solution(fit(new_model; start_val = start))
                @. total_sum         += sol
                @. total_squared_sum += sol^2
            catch
                n_failed[] += 1
            end
        end
    else
        n_threads = Threads.nthreads()
        # Pre-create one independent model copy per thread via deepcopy.
        model_pool = Channel(n_threads)
        for _ in 1:n_threads
            put!(model_pool, deepcopy(fitted.model))
        end
        # fit models in parallel
        lk = ReentrantLock()
        Threads.@threads for _ in 1:n_boot
            thread_model = take!(model_pool)
            try
                sample_data = bootstrap_sample(data)
                new_model = replace_observed(
                    thread_model;
                    data = sample_data, specification = specification, kwargs...)
                sol = solution(fit(new_model; start_val = start))
                lock(lk) do
                    @. total_sum         += sol
                    @. total_squared_sum += sol^2
                end
            catch
                lock(lk) do
                    n_failed[] += 1
                end
            finally
                put!(model_pool, thread_model)
            end
        end
    end
    # compute parameters
    n_conv = n_boot - n_failed[]
    sd = sqrt.(total_squared_sum / n_conv - (total_sum / n_conv) .^ 2)
    if !iszero(n_failed[])
        @warn "During bootstrap sampling, "*string(n_failed[])*" models did not converge"
    end
    return sd
end

function bootstrap_sample(data::Matrix)
    nobs = size(data, 1)
    index_new = rand(1:nobs, nobs)
    data_new = data[index_new, :]
    return data_new
end

bootstrap_sample(data::Dict) = Dict(k => bootstrap_sample(data[k]) for k in keys(data))

function prepare_data_bootstrap(data, model::AbstractSemSingle)
    if isnothing(data)
        data = samples(observed(model))
    end
    data = Matrix(data)
    return data
end

function prepare_data_bootstrap(data, model::SemEnsemble)
    sems = model.sems
    groups = model.groups
    if isnothing(data)
        data = Dict(g => samples(observed(m)) for (g, m) in zip(groups, sems))
    end
    data = Dict(k => Matrix(data[k]) for k in keys(data))
    return data
end


