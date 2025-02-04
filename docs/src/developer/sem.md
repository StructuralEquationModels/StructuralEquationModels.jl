# Custom model types

The abstract supertype for all models is `AbstractSem`, which has two subtypes, `AbstractSemSingle{O, I, L}` and `AbstractSemCollection`. Currently, there are 2 subtypes of `AbstractSemSingle`: `Sem`, `SemFiniteDiff`. All subtypes of `AbstractSemSingle` should have at least observed, implied, loss and optimizer fields, and share their types (`{O, I, L}`) with the parametric abstract supertype. For example, the `SemFiniteDiff` type is implemented as

```julia
struct SemFiniteDiff{O <: SemObserved, I <: SemImplied, L <: SemLoss} <:
       AbstractSemSingle{O, I, L}
    observed::O
    implied::I
    loss::L
end
```

Additionally, you can change how objective/gradient/hessian values are computed by providing methods for `evaluate!`, e.g. from `SemFiniteDiff`'s implementation:

```julia
evaluate!(objective, gradient, hessian, model::SemFiniteDiff, params) = ...
```

Additionally, we can define constructors like the one in `"src/frontend/specification/Sem.jl"`.

It is also possible to add new subtypes for `AbstractSemCollection`.