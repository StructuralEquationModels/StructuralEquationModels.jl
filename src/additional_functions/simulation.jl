############################################################################################
# simulate data
############################################################################################
"""
    rand(sem::Union{Sem, SemLoss, SemImplied}, [params], n)

Sample from the multivariate normal distribution implied by the SEM model.

# Arguments
- `sem`: SEM model to use. Ensemble models with multiple SEM terms are not supported.
- `params`: optional SEM model parameters to simulate from, otherwise uses the
  current state of implied covariances and means.
- `n::Integer`: Number of samples to draw.

# Examples
```julia
rand(model, start_simple(model), 100)
```
"""
function Distributions.rand(implied::SemImplied, params, n::Integer)
    if !isnothing(params)
        # update the implied covariances with the new model params
        update!(EvaluationTargets{true, false, false}(), implied, params)
    end
    Σ = Symmetric(implied.Σ)
    if MeanStruct(implied) === NoMeanStruct
        return permutedims(rand(MvNormal(Σ), n))
    elseif MeanStruct(implied) === HasMeanStruct
        return permutedims(rand(MvNormal(implied.μ, Σ), n))
    end
end

Distributions.rand(loss::SemLoss, params, n::Integer) = rand(SEM.implied(loss), params, n)

Distributions.rand(model::Sem, params, n::Integer) = rand(sem_term(model), params, n)

# rand() overloads without SEM params
Distributions.rand(implied::Union{SemImplied, SemLoss, Sem}, n::Integer) =
    Distributions.rand(implied, nothing, n)
