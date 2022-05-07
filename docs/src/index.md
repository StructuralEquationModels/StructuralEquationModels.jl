# StructuralEquationModes.jl: a fast and flexible SEM framework

This is a package for Structural Equation Modeling.
It is still *in development*.
Models you can fit include
- Linear SEM that can be specified in RAM notation
- ML, GLS and FIML estimation
- Ridge Regularization
- Multigroup SEM
- Sums of arbitrary loss functions (everything the optimizer can handle)

We provide fast objective functions, gradients, and for some cases hessians as well as approximations thereof.
As a user, you can easily define custom loss functions.
For those, you can decide to provide analytical gradients or use finite difference approximation / automatic differentiation.
You can choose to mix and match loss functions natively found in this package and those you provide.
In such cases, you optimize over a sum of different objectives (e.g. ML + Ridge).
This mix and match strategy also applies to gradients, where you may supply analytic gradients or opt for automatic differentiation or mix analytical and automatic differentiation.

You may consider using this package if:
- you want to extend SEM (e.g. add a new objective function) and need an extendable framework
- you want to extend SEM, and your implementation needs to be fast (because you want to do a simulation, for example)
- you want to fit the same model(s) to many datasets (bootstrapping, simulation studies)
- you are planning a study and would like to do power simulations

For examples on how to use the package, see the Tutorials.

## Installation

To install the latest version from GitHub, use the following commands in your julia REPL:
```julia
using Pkg
Pkg.add("https://github.com/StructuralEquationModels/StructuralEquationModels.jl")
```

## Citing the package

To cite our package, see [this page](https://github.com/StructuralEquationModels/StructuralEquationModels.jl/blob/main/CITATION.cff).
