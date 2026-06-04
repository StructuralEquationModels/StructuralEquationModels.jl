# Mixed analytical and automatic differentiation

This way of specifying our model is not ideal, however, because now also the maximum likelihood loss function lives inside a `SemFiniteDiff` model, and this means even though we have defined analytical gradients for it, we do not make use of them.

A more efficient way is therefore to specify our model as a combined model with multiple loss terms:

```julia
ml_term = SemML(
    SemObservedData(data = data, specification = partable),
    RAMSymbolic(partable)
)

ridge_penalty = SemRidge(partable)

model_ml_ridge = Sem(ml_term, ridge_penalty => 0.01)

model_ml_ridge_fit = fit(model_ml_ridge)
```

The results of both methods will be the same, but we can verify that the computation costs differ (the package `BenchmarkTools` has to be installed for this):

```julia
using BenchmarkTools

@benchmark fit(model)

@benchmark fit(model_ml_ridge)
```