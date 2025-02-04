# Custom implied types

We recommend to first read the part [Custom loss functions](@ref), as the overall implementation is the same and we will describe it here more briefly.

Implied types are of subtype `SemImplied`. To implement your own implied type, you should define a struct

```julia
struct MyImplied <: SemImplied
    ...
end
```

and a method to update!:

```julia
import StructuralEquationModels: objective!

function update!(targets::EvaluationTargets, implied::MyImplied, model::AbstractSemSingle, params)

    if is_objective_required(targets)
        ...
    end

    if is_gradient_required(targets)
        ...
    end
    if is_hessian_required(targets)
        ...
    end

end
```

As you can see, `update` gets passed as a first argument `targets`, which is telling us whether the objective value, gradient, and/or hessian are needed.
We can then use the functions `is_..._required` and conditional on what the optimizer needs, we can compute and store things we want to make available to the loss functions. For example, as we have seen in [Second example - maximum likelihood](@ref), the `RAM` implied type computes the model-implied covariance matrix and makes it available via `implied.Σ`.



Just as described in [Custom loss functions](@ref), you may define a constructor. Typically, this will depend on the `specification = ...` argument that can be a `ParameterTable` or a `RAMMatrices` object.

We implement an `ImpliedEmpty` type in our package that does nothing but serving as an `implied` field in case you are using a loss function that does not need any implied type at all. You may use it as a template for defining your own implied type, as it also shows how to handle the specification objects:

```julia
############################################################################################
### Types
############################################################################################
"""
Empty placeholder for models that don't need an implied part.
(For example, models that only regularize parameters.)

# Constructor

    ImpliedEmpty(;specification, kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object

# Examples
A multigroup model with ridge regularization could be specified as a `SemEnsemble` with one
model per group and an additional model with `ImpliedEmpty` and `SemRidge` for the regularization part.

# Extended help

## Interfaces
- `params(::RAMSymbolic) `-> Vector of parameter labels
- `nparams(::RAMSymbolic)` -> Number of parameters

## Implementation
Subtype of `SemImplied`.
"""
struct ImpliedEmpty{A, B, C} <: SemImplied
    hessianeval::A
    meanstruct::B
    ram_matrices::C
end

############################################################################################
### Constructors
############################################################################################

function ImpliedEmpty(;specification, meanstruct = NoMeanStruct(), hessianeval = ExactHessian(), kwargs...)
    return ImpliedEmpty(hessianeval, meanstruct, convert(RAMMatrices, specification))
end

############################################################################################
### methods
############################################################################################

update!(targets::EvaluationTargets, implied::ImpliedEmpty, par, model) = nothing

############################################################################################
### Recommended methods
############################################################################################

update_observed(implied::ImpliedEmpty, observed::SemObserved; kwargs...) = implied
```

As you see, similar to [Custom loss functions](@ref) we implement a method for `update_observed`.