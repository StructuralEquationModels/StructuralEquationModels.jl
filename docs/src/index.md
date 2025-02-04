# A fast and flexible SEM framework

`StructuralEquationModels.jl` is a package for Structural Equation Modeling (SEM) still under active development.
It is written for one purpose: Facilitating methodological innovations for SEM.
This purpose implies two subgoals for the package: Easy extensibility and speed.
You can easily define custom objective functions and other parts of the model.
At the same time, it is (very) fast.
These properties enable SEM researchers (such as you!) to play around with ideas (extensibility) and run extensive simulations (speed) to evaluate these ideas and users to profit from the resulting innovation.

## Get Started

To get started, we recommend the following order:

1. install the package ([Installation](@ref)),
2. read [A first model](@ref), and
3. get familiar with [Our Concept of a Structural Equation Model](@ref).

After that, if you are interested in specifying your own loss function (or other parts), you can proceed with [Extending the package](@ref).

## Target Group

You may consider using this package if you need **extensibility** and/or **speed**, e.g.
- you want to extend SEM (e.g. add a new objective function)
- you want to extend SEM, and your implementation needs to be fast
- you want to fit the same model(s) to many datasets (bootstrapping, simulation studies)
- you are planning a study and would like to do power simulations

For examples of how to use the package, see the Tutorials.

## Batteries Included

Models you can fit out of the box include
- Linear SEM that can be specified in RAM notation
- ML, GLS and FIML estimation
- Ridge/Lasso/... Regularization
- Multigroup SEM
- Sums of arbitrary loss functions (everything the optimizer can handle)

We provide fast objective functions, gradients, and for some cases hessians as well as approximations thereof.
As a user, you can easily define custom loss functions.
For those, you can decide to provide analytical gradients or use finite difference approximation / automatic differentiation.
You can choose to mix loss functions natively found in this package and those you provide.
In such cases, you optimize over a sum of different objectives (e.g. ML + Ridge).
This strategy also applies to gradients, where you may supply analytic gradients or opt for automatic differentiation or mixed analytical and automatic differentiation.

### Installation
You must have [julia](https://julialang.org/downloads/) installed (and we strongly recommend using an IDE of your choice; we like VS Code with the Julia extension).

To install the latest version of our package, use the following commands:

```julia
julia> ]
pkg> add StructuralEquationModels
```

## Citing the package

To cite our package, go to the [GitHub repostory](https://github.com/StructuralEquationModels/StructuralEquationModels.jl) and click on "Cite this repostiory" on the right side or see [the CSL file](https://github.com/StructuralEquationModels/StructuralEquationModels.jl/blob/main/CITATION.cff).
