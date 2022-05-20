# Custom loss functions

As an example, we will implement ridge regularization. For example, maximum likelihood estimation with ridge regularization consists of optimizing the objective
```math
F_{ML}(\theta) + \alpha \lVert \theta_I \rVert^2_2
```
Since we allow for the optimization of sums of loss functions, and the maximum likelihood loss function already exists, we only need to implement the ridge part (and get additionally ridge regularization for WLS and FIML estimation for free).

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

That's all we need to make it work! For example, we can now fit [A first model](@ref) now with ridge regularization:
```julia

```


## Convenient

To be able to build model with the [Outer Constructor](@ref), you need to add a constructor for your loss function that only takes keyword arguments and allows for passing optional additional kewyword arguments. A constructor is just a function that creates a new instance of your type:

```julia
function MyLoss(;arg1 = ..., arg2, kwargs...)
    ...
    return MyLoss(...)
end
```