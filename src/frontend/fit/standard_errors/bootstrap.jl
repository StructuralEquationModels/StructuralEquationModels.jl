# base type for accumulators of intermediate bootstrap results
abstract type BootstrapAccumulator end

# internal function to run bootstrap
function bootstrap!(
    acc::BootstrapAccumulator,
    fitted::SemFit;
    data = nothing,
    engine = :Optim,
    parallel = false,
    fit_kwargs = Dict(),
)
    sem = model(fitted)
    data = isnothing(data) ? _bootstrap_data(sem) : data
    start = solution(fitted)

    n_boot = n_bootstrap(acc)

    # fit to bootstrap samples
    if !parallel
        bs_sem = deepcopy(sem) # avoid mutating the original model
        for i in 1:n_boot
            new_fit = _fit_bootstrap_sample(bs_sem, data, start; engine, fit_kwargs)
            update!(acc, i, new_fit, nothing)
        end
    else
        n_threads = Threads.nthreads()
        # Pre-create one independent model copy per thread via deepcopy.
        model_pool = Channel(n_threads)
        for _ in 1:n_threads
            put!(model_pool, deepcopy(sem))
        end
        lk = ReentrantLock()
        Threads.@threads for i in 1:n_boot
            thread_model = take!(model_pool)
            new_fit = _fit_bootstrap_sample(thread_model, data, start; engine, fit_kwargs)
            update!(acc, i, new_fit, lk)
            put!(model_pool, thread_model)
        end
    end

    return acc
end

# a simple accumulator that just stores the statistic for each sample and whether it converged
struct SimpleBootstrapAccumulator{F} <: BootstrapAccumulator
    statistic::F
    samples::Vector{Any}
    converged_mask::Vector{Bool}
end

SimpleBootstrapAccumulator(statistic, n_boot::Integer) =
    SimpleBootstrapAccumulator(statistic, Vector{Any}(undef, n_boot), fill(false, n_boot))

n_bootstrap(acc::SimpleBootstrapAccumulator) = length(acc.samples)

function update!(acc::SimpleBootstrapAccumulator, i::Integer, fit::SemFit, _)
    acc.samples[i] = acc.statistic(fit)
    acc.converged_mask[i] = converged(fit)
end

"""
   struct BootstrapResult{T}

Stores the output of a [`bootstrap`](@ref) call.
"""
struct BootstrapResult{T}
    samples::Vector{T}
    converged_mask::BitVector
    n_boot::Int
    n_converged::Int
end

function Base.show(io::IO, result::BootstrapResult{T}) where {T}
    println(
        io,
        "BootstrapResult{$(T)} with $(result.n_converged)/$(result.n_boot) converged samples",
    )
end

"""
    bootstrap(
        fitted::SemFit;
        statistic = solution,
        n_boot = 3000,
        data = nothing,
        engine = :Optim,
        parallel = false,
        fit_kwargs = Dict()
    ) -> BootstrapResult

Bootstrap the samples and apply `statistic` function to each.

Returns a [`BootstrapResult`](@ref) object containing the results of `statistic`
applied to each bootstrapped sample.

Supports both single-group and multi-group models.
For multi-group models, each group is resampled independently.

# Arguments
- `fitted`: a fitted SEM.
- `statistic`: any function that can be called on a `SemFit` object.
  The output will be returned as the bootstrap sample.
- `n_boot`: number of boostrap samples
- `data`: data to sample from. Only needed if different than the fitted model.
  For multi-group models, pass a `Dict{Symbol}` mapping term ids to data matrices.
- `engine`: optimizer engine, passed to `fit`.
- `parallel`: if `true`, run bootstrap samples in parallel on all available threads.
  The number of threads is controlled by the `JULIA_NUM_THREADS` environment variable or
  the `--threads` flag when starting Julia.
- `fit_kwargs` : a `Dict` controlling model fitting for each bootstrap sample,
  passed to [`fit`](@ref)

# Example
```julia
# 1000 bootstrap samples of the minimum, fitted with :Optim
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
    statistic = solution;
    n_boot = 3000,
    data = nothing,
    engine = :Optim,
    parallel = false,
    fit_kwargs = Dict(),
)
    acc = SimpleBootstrapAccumulator(statistic, n_boot)
    bootstrap!(acc, fitted; data, engine, parallel, fit_kwargs)
    return BootstrapResult(
        [s for s in acc.samples],
        convert(BitVector, acc.converged_mask),
        n_bootstrap(acc),
        sum(acc.converged_mask),
    )
end

# bootstrap accumulator for se_bootstrap()
# accumulates per-parameter sum and sum of squares across bootstrap samples
struct StdErrBootstrapAccumulator <: BootstrapAccumulator
    n_boot::Int
    sum::Vector{Float64}
    squared_sum::Vector{Float64}
    n_converged::Ref{Int}
end

n_bootstrap(acc::StdErrBootstrapAccumulator) = acc.n_boot

StdErrBootstrapAccumulator(n_params::Integer, n_boot::Integer) =
    StdErrBootstrapAccumulator(n_boot, zeros(n_params), zeros(n_params), Ref(0))

function update!(
    acc::StdErrBootstrapAccumulator,
    i::Integer,
    fit::SemFit,
    lk::Union{Base.AbstractLock, Nothing},
)
    conv = converged(fit)
    if conv
        sol = solution(fit)
        isnothing(lk) || lock(lk)
        acc.n_converged[] += 1
        @. acc.sum += sol
        @. acc.squared_sum += abs2(sol)
        isnothing(lk) || unlock(lk)
    end
end

"""
    se_bootstrap(fitted::SemFit; n_boot = 3000, kwargs...)

Calculate standard errors using bootstrap approach.

Supports both single-group and multi-group models.
For multi-group models, each group is resampled independently.

# Arguments
- `fitted`: a fitted SEM.
- `n_boot`: number of boostrap samples
- `data`: data to sample from. Only needed if different than the fitted model.
  For multi-group models, pass a `Dict{Symbol}` mapping term ids to data matrices.
- `engine`: optimizer engine, passed to `fit`.
- `parallel`: if `true`, run bootstrap samples in parallel on all available threads.
  The number of threads is controlled by the `JULIA_NUM_THREADS` environment variable or
  the `--threads` flag when starting Julia.
- `fit_kwargs` : a `Dict` controlling model fitting for each bootstrap sample,
  passed to [`fit`](@ref)


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
function se_bootstrap(fitted::SemFit; n_boot = 3000, kwargs...)
    acc = StdErrBootstrapAccumulator(nparams(fitted), n_boot)
    bootstrap!(acc, fitted; kwargs...)
    n_conv = acc.n_converged[]
    @info "$n_conv models converged"

    if n_conv == 0
        @warn "No bootstrap samples converged. Returning NaN."
        return fill(NaN, length(acc.sum))
    else
        return sqrt.(acc.squared_sum ./ n_conv - abs2.(acc.sum / n_conv))
    end
end

############################################################################################
### Helper Functions
############################################################################################

"""
    resample_with_replacement(data::AbstractMatrix)
    resample_with_replacement(data::AbstractVector{<:AbstractMatrix})

Resample rows of a data matrix with replacement (bootstrap sample).
For a vector of matrices (multi-group models), independently resamples each matrix.
"""
function resample_with_replacement(data::AbstractMatrix)
    n = size(data, 1)
    return data[rand(1:n, n), :]
end

function resample_with_replacement(data::AbstractVector{<:AbstractMatrix})
    return [resample_with_replacement(term_data) for term_data in data]
end

# Extract data from a model for bootstrap resampling.
function _bootstrap_data(sem::AbstractSem)
    terms = sem_terms(sem)
    if length(terms) == 1
        return samples(observed(loss(terms[1])))
    else
        return [samples(observed(loss(term))) for term in terms]
    end
end

# Fit one bootstrap replicate: resample, replace observed data, fit.
function _fit_bootstrap_sample(sem_model, data, start; engine, fit_kwargs)
    boot_data = resample_with_replacement(data)
    # we replace the observed data with the bootstrapped one,
    # but preserve any internal state that is associated with the original data
    boot_model = replace_observed(sem_model, boot_data; recompute_observed_state = true)
    return fit(boot_model; start_val = start, engine = engine, fit_kwargs...)
end
