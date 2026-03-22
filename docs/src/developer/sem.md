# Custom model types

The abstract supertype for all models is [`AbstractSem`](@ref). Currently, there are 2 concrete subtypes:
`Sem{L <: Tuple}` and `SemFiniteDiff{S <: AbstractSem}`.
A `Sem` model holds a tuple of `LossTerm`s (each wrapping an `AbstractLoss`) and a vector of parameter labels. Both single-group and multigroup models are represented as `Sem`.

`SemFiniteDiff` wraps any `AbstractSem` and substitutes dedicated gradient/hessian evaluation with finite difference approximation:

```julia
struct SemFiniteDiff{S <: AbstractSem} <: AbstractSem
    model::S
end
```

Additionally, you can change how objective/gradient/hessian values are computed by providing methods for `evaluate!`, e.g. from `SemFiniteDiff`'s implementation:

```julia
evaluate!(objective, gradient, hessian, model::SemFiniteDiff, params) = ...
```

Additionally, we can define constructors like the one in `"src/frontend/specification/Sem.jl"`.