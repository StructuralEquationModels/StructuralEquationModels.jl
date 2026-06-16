# Custom loss functions

As an example, we will implement ridge regularization. Maximum likelihood estimation with ridge regularization consists of optimizing the objective
```math
F_{ML}(\theta) + \alpha \lVert \theta_I \rVert^2_2
```
Since we allow for the optimization of sums of loss functions, and the maximum likelihood loss function already exists, we only need to implement the ridge part (and additionally get ridge regularization for WLS and FIML estimation for free).

## Minimal
```@setup loss
using StructuralEquationModels
```

To define a new loss function, you have to define a new type that is a subtype of `AbstractLoss`:
```@example loss
struct MyRidge <: AbstractLoss
    α
    I
end
```
We store the hyperparameter α and the indices I of the parameters we want to regularize.

Additionaly, we need to define a *method* of the function `evaluate!` to compute the objective:

```@example loss
import StructuralEquationModels: evaluate!

evaluate!(objective::Number, gradient::Nothing, hessian::Nothing, ridge::MyRidge, par) =
    ridge.α * sum(i -> abs2(par[i]), ridge.I)
```

The function `evaluate!` recognizes by the types of the arguments `objective`, `gradient` and `hessian` whether it should compute the objective value, gradient or hessian of the model w.r.t. the parameters.
In this case, `gradient` and `hessian` are of type `Nothing`, signifying that they should not be computed, but only the objective value.

That's all we need to make it work! For example, we can now fit [A first model](@ref) with ridge regularization:

We first give some parameters labels to be able to identify them as targets for the regularization:

```@example loss
observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

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
    graph,
    latent_vars = latent_vars,
    observed_vars = observed_vars
)

parameter_indices = getindex.([param_indices(partable)], [:a, :b, :c])
myridge = MyRidge(0.01, parameter_indices)

model = SemFiniteDiff(
    specification = partable,
    data = example_data("political_democracy"),
    loss = (SemML, myridge)
)

model_fit = fit(model)
```

This is one way of specifying the model - we now have **one model** with **multiple loss functions**. Because we did not provide a gradient for `MyRidge`, we have to specify a `SemFiniteDiff` model that computes numerical gradients with finite difference approximation.

Ridge regularization only depends on the parameters, so the `evaluate!` method above does not need anything else. Other loss functions, however, depend on the observed data and on what the model implies about it. Loss functions that compare the implied and the observed structure are subtypes of [`SemLoss`](@ref) and store their own `observed` and `implied` parts, which can be accessed inside `evaluate!` via `observed(loss)` and `implied(loss)`. See [Second example - maximum likelihood](@ref) for information on how to do that.

### Improve performance

By far the biggest improvements in performance will result from specifying analytical gradients. We can do this for our example:

```@example loss
function evaluate!(objective, gradient, hessian::Nothing, ridge::MyRidge, par)
    # compute gradient
    if !isnothing(gradient)
        fill!(gradient, 0)
        gradient[ridge.I] .= 2 * ridge.α * par[ridge.I]
    end
    # compute objective
    if !isnothing(objective)
        return ridge.α * sum(i -> par[i]^2, ridge.I)
    end
end
```

As you can see, in this method definition, both `objective` and `gradient` can be different from `nothing`.
We then check whether to compute the objective value and/or the gradient with `isnothing(objective)`/`isnothing(gradient)`.
This syntax makes it possible to compute objective value and gradient at the same time, which is beneficial when the the objective and gradient share common computations.

Now, instead of specifying a `SemFiniteDiff`, we can use the normal `Sem` constructor:

```@example loss
model_new = Sem(
    specification = partable,
    data = example_data("political_democracy"),
    loss = (SemML, myridge)
)

model_fit = fit(model_new)
```

The results are the same, but we can verify that the computational costs are way lower (for this, the julia package `BenchmarkTools` has to be installed):

```julia
using BenchmarkTools

@benchmark fit(model)

@benchmark fit(model_new)
```

The exact results of those benchmarks are of course highly depended an your system (processor, RAM, etc.), but you should see that the median computation time with analytical gradients drops to about 5% of the computation without analytical gradients.

Additionally, you may provide analytic hessians by writing a respective method for `evaluate!`. However, this will only matter if you use an optimization algorithm that makes use of the hessians. Our default algorithmn `LBFGS` from the package `Optim.jl` does not use hessians (for example, the `Newton` algorithmn from the same package does).

## Convenient

To be able to build the loss term, it needs a constructor.
Every `SemLoss` subtype should provide a constructor with 3 positional arguments:
  * `observed::SemObserved`: the observed part of the model
  * `implied::SemImplied`: the implied part of the model
  * `refloss::Union{MyLoss, Nothing} = nothing`: optional loss term of the same type
    to use as a reference for any loss-specific configuration.

Any additional loss configuration details should be passed as optional keyword arguments.
If both `refloss` and the keyword arguments are provided, the keyword arguments take
precedence. This constructor is used internally by the functions like [`replace_observed`](@ref)
to rebuild the loss term with new observed data while preserving the implied state.

```julia
function MyLoss(
    observed::SemObserved, implied::SemImplied, refloss::Union{MyLoss, Nothing} = nothing;
    kwarg1 = ..., kwarg2 = ..., kwargs...
)
    ...
    return MyLoss(...) # internal MyLoss constructor
end
```

## Additional functionality

### Access additional information

If you want to provide a way to query information about loss functions of your type, you can provide functions for that:

```julia
hyperparameter(ridge::MyRidge) = ridge.α
regularization_indices(ridge::MyRidge) = ridge.I
```

# Second example - maximum likelihood
Let's make a sligtly more complicated example: we will reimplement maximum likelihood estimation.

To keep it simple, we only cover models without a meanstructure. The maximum likelihood objective is defined as

```math
F_{ML} = \log \det \Sigma_i + \mathrm{tr}\left(\Sigma_{i}^{-1} \Sigma_o \right)
```

where ``\Sigma_i`` is the model implied covariance matrix and ``\Sigma_o`` is the observed covariance matrix. We can query the model implied covariance matrix from the `implied` part of our loss term, and the observed covariance matrix from the `observed` part of our loss term.

Since this loss function compares the implied and the observed structure, it is a subtype of [`SemLoss`](@ref) rather than a plain `AbstractLoss`. Every `SemLoss` stores its own `observed` and `implied` parts, which can be accessed inside `evaluate!` via `observed(loss)` and `implied(loss)`.

To get information on what we can access from a certain `implied` or `observed` type, we can check it`s documentation an the pages [API - model parts](@ref) or via the help mode of the REPL:

```julia
julia>?

help?> RAM

help?> SemObservedData
```

We see that the model implied covariance matrix can be assessed as `implied(loss).Σ` and the observed covariance matrix as `obs_cov(observed(loss))`.

A `SemLoss` subtype stores its `observed` and `implied` parts in the first two fields, and provides a constructor with the positional arguments `(observed, implied, refloss = nothing; kwargs...)` (see the [Convenient](@ref) section above). This constructor is used by the [`Sem`](@ref) constructor to build the loss term. With this information, we can implement maximum likelihood optimization as

```@example loss
struct MaximumLikelihood{O <: SemObserved, I <: SemImplied} <: SemLoss{O, I}
    observed::O
    implied::I
end

# constructor used by the `Sem` constructor to build the loss term
MaximumLikelihood(observed::SemObserved, implied::SemImplied, refloss = nothing; kwargs...) =
    MaximumLikelihood{typeof(observed), typeof(implied)}(observed, implied)

using LinearAlgebra
import StructuralEquationModels: evaluate!

function evaluate!(objective::Number, gradient::Nothing, hessian::Nothing, semml::MaximumLikelihood, par)
    # access the model implied and observed covariance matrices
    Σᵢ = implied(semml).Σ
    Σₒ = obs_cov(observed(semml))
    # compute the objective
    if isposdef(Symmetric(Σᵢ)) # is the model implied covariance matrix positive definite?
        return logdet(Σᵢ) + tr(inv(Σᵢ)*Σₒ)
    else
        return Inf
    end
end
```

to deal with eventual non-positive definiteness of the model implied covariance matrix, we chose the pragmatic way of returning infinity whenever this is the case.

Let's specify and fit a model:

```@example loss
model_ml = SemFiniteDiff(
    specification = partable,
    data = example_data("political_democracy"),
    loss = MaximumLikelihood
)

model_fit = fit(model_ml)
```

If you want to differentiate your own loss functions via automatic differentiation, check out the [AutoDiffSEM](https://github.com/StructuralEquationModels/AutoDiffSEM) package.
