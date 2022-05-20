# Custom loss functions

As an example, we will implement ridge regularization. Maximum likelihood estimation with ridge regularization consists of optimizing the objective
```math
F_{ML}(\theta) + \alpha \lVert \theta_I \rVert^2_2
```
Since we allow for the optimization of sums of loss functions, and the maximum likelihood loss function already exists, we only need to implement the ridge part (and additionally get ridge regularization for WLS and FIML estimation for free).

## Minimal
To define a new loss function, you have to define a new type that is a subtype of `SemLossFunction`:
```julia
struct Ridge <: SemLossFunction
    α
    I
end
```
We store the hyperparameter α and the indices I of the parameters we want to regularize.

Additionaly, we need to define a *method* to compute the objective:

```julia
import StructuralEquationModels: objective!

objective!(lossfun::Ridge, par, model::AbstractSemSingle) = ridge.α*sum(par[I].^2)
```

That's all we need to make it work! For example, we can now fit [A first model](@ref) with ridge regularization:

We first give som eparameters labels to be able to identify as targets for the regularization:
```julia
graph = @StenoGraph begin

    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8

    # latent regressions
    ind60 → label(:a)*dem60
    dem60 → label(:b)*dem65
    ind60 → label(:c)*dem65

    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6

end

partable = ParameterTable(
    latent_vars = latent_vars,
    observed_vars = observed_vars,
    graph = graph)
```

```julia
parameter_indices  = get_identifier_indices([:a, :b, :c], partable)
myridge = Ridge(0.01, parameter_indices)

model = SemFiniteDiff(
    specification = partable,
    data = data,
    loss = (SemML, myridge)
)

model_fit = sem_fit(model)
```

This is one way of specifying the model - we now have **one model** with **multiple loss functions**. Because we did not provide a gradient for `Ridge`, we have to specify a `SemFiniteDiff` model that computes numerical gradients with finite difference approximation.

This way of specifying our model is not ideal, however, because now also the maximum likelihood loss function lives inside a `SemFiniteDiff` model, and this means even though we have defined analytical gradients for it, we do not make use of them.

A more efficient way is therefore to specify our model as an ensemble model: 

```julia
model_ml = Sem(
    specification = partable,
    data = data,
    loss = SemML
)

model_ridge = SemFiniteDiff(
    specification = partable,
    data = data,
    loss = myridge
)


model_ml_ridge = SemEnsemble(model_ml, model_ridge)

model_ml_ridge_fit = sem_fit(model_ml_ridge)
```

The results of both methods will be the same, but we can verify that the computation costs differ (the package `BenchmarkTools` has to be installed for this):

```julia
using BenchmarkTools

@benchmark sem_fit(model)

@benchmark sem_fit(model_ml_ridge)
```

## Convenient

To be able to build model with the [Outer Constructor](@ref), you need to add a constructor for your loss function that only takes keyword arguments and allows for passing optional additional kewyword arguments. A constructor is just a function that creates a new instance of your type:

```julia
function MyLoss(;arg1 = ..., arg2, kwargs...)
    ...
    return MyLoss(...)
end
```