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

objective!(ridge::Ridge, par, model::AbstractSemSingle) = ridge.α*sum(par[ridge.I].^2)
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

### Improve performance

By far the biggest improvements in performance will result from specifying analytical gradients. We can do this for our example:

```julia
import StructuralEquationModels: gradient!

function gradient!(ridge::Ridge, par, model::AbstractSemSingle)
    gradient = zero(par)
    gradient[ridge.I] .= 2*ridge.α*par[ridge.I]
    return gradient
end
```

Now, instead of specifying a `SemFiniteDiff`, we can use the normal `Sem` constructor:

```julia
model_new = Sem(
    specification = partable,
    data = data,
    loss = (SemML, myridge)
)

model_fit = sem_fit(model_new)
```

The results are thew same, but we can verify that the computational costs are way lower (for this, the julia package `BenchmarkTools` has to be installed):

```julia
using BenchmarkTools

@benchmark sem_fit(model)

@benchmark sem_fit(model_new)
```

The exact results of those benchmarks are of course highly depended an your system (processor, RAM, etc.), but you should see that the median computation time with analytical gradients drops to about 5% of the computation without analytical gradients.

Additionally, you may provide analytic hessians by writing a method of the form

```julia
function hessian!(ridge::Ridge, par, model::AbstractSemSingle)
    ...
    return hessian
end
```

however, this will only matter if you use an optimization algorithm that makes use of the hessians. Our gefault algorithmn `LBFGS` from the package `Optim.jl` does not use hessians (for example, the `Newton` algorithmn from the same package does).

Do improve performance even more, you can write a method of the form

```julia
function objective_gradient!(ridge::Ridge, par, model::AbstractSemSingle)
    ...
    return objective, gradient
end
```

This is beneficial when the computation of the objective and gradient share common computations. For example, in maximum likelihood estimation, the model implied covariance matrix has to be inverted to both compute the objective and gradient. Whenever the optimization algorithmn asks for the objective value and gradient at the same point, we call `objective_gradient!` and only have to do the shared computations - in this case the matric inversion - once.

If you want to do hessian-based optimization, there are also the following methods:

```julia
function objective_hessian!(ridge::Ridge, par, model::AbstractSemSingle)
    ...
    return objective, hessian
end

function gradient_hessian!(ridge::Ridge, par, model::AbstractSemSingle)
    ...
    return gradient, hessian
end

function objective_gradient_hessian!(ridge::Ridge, par, model::AbstractSemSingle)
    ...
    return objective, gradient, hessian
end
```

## Convenient

To be able to build the model with the [Outer Constructor](@ref), you need to add a constructor for your loss function that only takes keyword arguments and allows for passing optional additional kewyword arguments. A constructor is just a function that creates a new instance of your type:

```julia
function MyLoss(;arg1 = ..., arg2, kwargs...)
    ...
    return MyLoss(...)
end
```

## Additional functionality

### Update observed data

If you are planing a simulation study where you have to fit the **same model** to many **different datasets**, it is computationally beneficial to not build the whole model completely new everytime you change your data.
Therefore, we provide a function to update the data of your model, `swap_observed(model(semfit); data = new_data)`. However, we can not now beforehand in what way your loss function depends on the specific datasets. The solution is to provide a method for `update_observed`. Since `Ridge` does not depend on the data at all, this is quite easy:

```julia
import StructuralEquationModels: update_observed

update_observed(ridge::Ridge, observed::SemObs; kwargs...) = ridge
```

### Access additional information

If you want to provide a way to query information about loss functions of your type, you can provide functions for that:

```julia
hyperparameter(ridge::Ridge) = ridge.α
regularization_indices(ridge::Ridge) = ridge.I
```