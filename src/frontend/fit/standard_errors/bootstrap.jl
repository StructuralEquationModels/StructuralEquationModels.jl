"""
    bootstrap(
        fitted::SemFit,
        specification::SemSpecification;
        statistic = solution,
        n_boot = 3000,
        data = nothing,
        engine = :Optim,
        parallel = false,
        fit_kwargs = Dict(),
        replace_kwargs = Dict())

Return bootstrap samples for `statistic`.

# Arguments
- `fitted`: a fitted SEM.
- `specification`: a `ParameterTable` or `RAMMatrices` object passed to `replace_observed`.
- `statistic`: any function that can be called on a `SemFit` object.
  The output will be returned as the bootstrap sample.
- `n_boot`: number of boostrap samples
- `data`: data to sample from. Only needed if different than the data from `sem_fit`
- `engine`: optimizer engine, passed to `fit`.
- `parallel`: if `true`, run bootstrap samples in parallel on all available threads.
  The number of threads is controlled by the `JULIA_NUM_THREADS` environment variable or
  the `--threads` flag when starting Julia.
- `fit_kwargs` : a `Dict` controlling model fitting for each bootstrap sample,
  passed to `fit`
- `replace_kwargs`: a `Dict` passed to `replace_observed`

# Example
```julia
# 1000 boostrap samples of the minimum, fitted with :Optim
bootstrap(
    fitted;
    statistic = StructuralEquationModels.minimum,
    n_boot = 1000,
    engine = :Optim,
)
```
"""
function bootstrap(
    fitted::SemFit,
    specification::SemSpecification;
    statistic = solution,
    n_boot = 3000,
    data = nothing,
    engine = :Optim,
    parallel = false,
    fit_kwargs = Dict(),
    replace_kwargs = Dict(),
)
    # access data and convert to matrix
    data = prepare_data_bootstrap(data, fitted.model)
    start = solution(fitted)
    # pre-allocations
    out = []
    conv = []
    errors = []
    n_failed = Ref(0)
    # fit to bootstrap samples
    if !parallel
        for _ in 1:n_boot
            try
                sample_data = bootstrap_sample(data)
                new_model = replace_observed(
                    fitted.model;
                    data = sample_data,
                    specification = specification,
                    replace_kwargs...,
                )
                new_fit = fit(new_model; start_val = start, engine = engine, fit_kwargs...)
                sample = statistic(new_fit)
                c = converged(new_fit)
                push!(out, sample)
                push!(conv, c)
            catch e
                n_failed[] += 1
                push!(errors, e)
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
                    data = sample_data,
                    specification = specification,
                    replace_kwargs...,
                )
                new_fit = fit(new_model; start_val = start, engine = engine, fit_kwargs...)
                sample = statistic(new_fit)
                c = converged(new_fit)
                lock(lk) do
                    push!(out, sample)
                    push!(conv, c)
                end
            catch e
                lock(lk) do
                    n_failed[] += 1
                    push!(errors, e)
                end
            finally
                put!(model_pool, thread_model)
            end
        end
    end
    # compute parameters
    if !iszero(n_failed[])
        @warn "During bootstrap sampling, "*string(n_failed[])*" samples errored."
    end
    return Dict(
        :samples => out,
        :n_boot => n_boot,
        :n_converged => isempty(conv) ? 0 : sum(conv),
        :converged => conv,
        :n_errored => n_failed[],
        :errors => errors
    )
end

"""
    se_bootstrap(
        fitted::SemFit,
        specification::SemSpecification;
        n_boot = 3000,
        data = nothing,
        parallel = false,
        fit_kwargs = Dict(),
        replace_kwargs = Dict())

Return bootstrap standard errors.

# Arguments
- `fitted`: a fitted SEM.
- `specification`: a `ParameterTable` or `RAMMatrices` object passed to `replace_observed`.
- `n_boot`: number of boostrap samples
- `data`: data to sample from. Only needed if different than the data from `sem_fit`
- `engine`: optimizer engine, passed to `fit`.
- `parallel`: if `true`, run bootstrap samples in parallel on all available threads.
  The number of threads is controlled by the `JULIA_NUM_THREADS` environment variable or
  the `--threads` flag when starting Julia.
- `fit_kwargs` : a `Dict` controlling model fitting for each bootstrap sample, 
  passed to `sem_fit`
- `replace_kwargs`: a `Dict` passed to `replace_observed`

# Example
```julia
# 1000 boostrap samples, fitted with :NLopt
using NLopt

se_bootstrap(
    fitted;
    n_boot = 1000,
    engine = :NLopt,
)
```
"""
function se_bootstrap(
    fitted::SemFit,
    specification::SemSpecification;
    n_boot = 3000,
    data = nothing,
    engine = :Optim,
    parallel = false,
    fit_kwargs = Dict(),
    replace_kwargs = Dict(),
)
    # access data and convert to matrix
    data = prepare_data_bootstrap(data, fitted.model)
    start = solution(fitted)
    # pre-allocations
    total_sum         = zero(start)
    total_squared_sum = zero(start)
    n_failed          = Ref(0)
    n_conv            = Ref(0)
    # fit to bootstrap samples
    if !parallel
        for _ in 1:n_boot
            try
                sample_data = bootstrap_sample(data)
                new_model = replace_observed(
                    fitted.model;
                    data = sample_data,
                    specification = specification,
                    replace_kwargs...,
                )
                new_fit = fit(new_model; start_val = start, engine = engine, fit_kwargs...)
                sol = solution(new_fit)
                conv = converged(new_fit)
                if conv
                    n_conv[]             += 1
                    @. total_sum         += sol
                    @. total_squared_sum += sol^2
                end
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
                    data = sample_data,
                    specification = specification,
                    replace_kwargs...,
                )
                new_fit = fit(new_model; start_val = start, engine = engine, fit_kwargs...)
                sol = solution(new_fit)
                conv = converged(new_fit)
                if conv
                    lock(lk) do
                        n_conv[]             += 1
                        @. total_sum         += sol
                        @. total_squared_sum += sol^2
                    end
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
    n_conv = n_conv[]
    sd = sqrt.(total_squared_sum / n_conv - (total_sum / n_conv) .^ 2)
    if !iszero(n_failed[])
        @warn "During bootstrap sampling, "*string(n_failed[])*" samples errored"
    end
    @info string(n_conv)*" models converged"
    return sd
end

############################################################################################
### Helper Functions
############################################################################################

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


