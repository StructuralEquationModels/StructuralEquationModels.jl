# Collections

With StructuralEquationModels.jl, you can fit weighted sums of structural equation models. 
The most common use case for this are [Multigroup models](@ref). 
Another use case may be optimizing the sum of loss functions for some of which you do know the analytic gradient, but not for others. 
In this case, you can optimize the sum of a `Sem` and a `SemFiniteDiff` (or any other differentiation method).

To use this feature, you have to construct a `SemEnsemble` model, which is actually quite easy:

```julia
# models
model_1 = Sem(...)

model_2 = SemFiniteDiff(...)

model_3 = Sem(...)

model_ensemble = SemEnsemble(model_1, model_2, model_3; optimizer = ...)
```

So you just construct the individual models (however you like) and pass them to `SemEnsemble`.
One important thing to note is that the individual optimizer entries of each model do not matter (as you can optimize your ensemble model only with one algorithmn from one optimization suite). Instead, `SemEnsemble` has its own optimizer part that specifies the backend for the whole ensemble model.
You may also pass a vector of weigths to `SemEnsemble`. By default, those are set to ``N_{model}/N_{total}``, i.e. each model is weighted by the number of observations in it's data (which matches the formula for multigroup models).

Multigroup models can also be specified via the graph interface; for an example, see [Multigroup models](@ref).

# API - collections

```@docs
SemEnsemble
AbstractSemCollection
```