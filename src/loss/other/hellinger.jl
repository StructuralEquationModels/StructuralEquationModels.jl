############################################################################################
### Types
############################################################################################
"""
Squared Hellinger distance loss. 

# Constructor

    Hellinger()

# Examples
```julia
    my_hellinger = Hellinger()
```

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""

struct Hellinger <: SemLossFunction end

using LinearAlgebra
import StructuralEquationModels: Σ, μ, obs_cov, obs_mean, objective!

function objective!(hell::Hellinger, parameters, model::AbstractSem) 

    # get model-implied covariance matrices
    Σᵢ = Σ(imply(model))
    
    # get observed covariance matrix
    Σₒ = obs_cov(observed(model))
    
    # get model-implued mean vector
    μᵢ = μ(imply(model))

    # get observed mean vector
    μₒ = obs_mean(observed(model))
    
    Sig = (Σᵢ + Σₒ)/2


    
    # compute the objective
    if isposdef(Symmetric(Σᵢ)) # is the model implied covariance matrix positive definite?

          loss = ( det(Σᵢ)^(1/4) * det(Σₒ)^(1/4) ) / sqrt(det(Sig)) 

           if !isnothing(μᵢ) & !isnothing(μₒ)
            μd = (μᵢ-μₒ)
            loss = loss * exp(0.125 * (μd)*inv(Sig)*(μd)')
          end
          
          loss = 1 - loss

          return loss

    else
        return Inf
    end
end