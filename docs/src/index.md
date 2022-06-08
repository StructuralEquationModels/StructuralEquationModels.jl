# A fast and flexible SEM framework

This is a package for Structural Equation Modeling *in development*.
It is written for extensibility, that is, you can easily define your own objective functions and other parts of the model.
At the same time, it is (very) fast.

To get started right away, we recommend installing our package ([Installation](@ref)) and then reading [A first model](@ref) and [Our Concept of a Structural Equation Model](@ref).

After that, if you are interested in specifying your own loss functions or other parts, you can proceed with [Extending the package](@ref).

Models you can fit out of the box include
- Linear SEM that can be specified in RAM notation
- ML, GLS and FIML estimation
- Ridge Regularization
- Multigroup SEM
- Sums of arbitrary loss functions (everything the optimizer can handle)

We provide fast objective functions, gradients, and for some cases hessians as well as approximations thereof.
As a user, you can easily define custom loss functions.
For those, you can decide to provide analytical gradients or use finite difference approximation / automatic differentiation.
You can choose to mix loss functions natively found in this package and those you provide.
In such cases, you optimize over a sum of different objectives (e.g. ML + Ridge).
This strategy also applies to gradients, where you may supply analytic gradients or opt for automatic differentiation or mixed analytical and automatic differentiation.

You may consider using this package if you need **extensibility** and/or **speed**, e.g.
- you want to extend SEM (e.g. add a new objective function)
- you want to extend SEM, and your implementation needs to be fast
- you want to fit the same model(s) to many datasets (bootstrapping, simulation studies)
- you are planning a study and would like to do power simulations

For examples on how to use the package, see the Tutorials.

## Installation
You need to have [julia](https://julialang.org/downloads/) installed (and we strongly recommend to additionally use an IDE of your choice; we like VS Code with the Julia extension).

To install the latest version of our package from GitHub, use the following commands:
```julia
using Pkg
Pkg.add(url = "https://github.com/StructuralEquationModels/StructuralEquationModels.jl")
```

## Citing the package

To cite our package, see [this page](https://github.com/StructuralEquationModels/StructuralEquationModels.jl/blob/main/CITATION.cff).