# <img src="https://github.com/StructuralEquationModels/Data/blob/main/images/logo.png" width = 100> StructuralEquationModels.jl

| **Documentation**                                                               | **Build Status**                                                                                | Citation                                                                                        |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://structuralequationmodels.github.io/StructuralEquationModels.jl/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://structuralequationmodels.github.io/StructuralEquationModels.jl/dev/) | [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![Github Action CI](https://github.com/StructuralEquationModels/StructuralEquationModels.jl/workflows/CI_extended/badge.svg)](https://github.com/StructuralEquationModels/StructuralEquationModels.jl/actions/) [![codecov](https://codecov.io/gh/StructuralEquationModels/StructuralEquationModels.jl/branch/main/graph/badge.svg?token=P2kjzpvM4V)](https://codecov.io/gh/StructuralEquationModels/StructuralEquationModels.jl) | [![DOI](https://zenodo.org/badge/228649704.svg)](https://zenodo.org/badge/latestdoi/228649704) |

# What is this Package for?

This is a package for Structural Equation Modeling.
It is still *in development*.
Models you can fit include
- Linear SEM that can be specified in RAM (or LISREL) notation
- ML, GLS and FIML estimation
- Regularization
- Multigroup SEM
- Sums of arbitrary loss functions (everything the optimizer can handle).

# What are the merrits?

We provide fast objective functions, gradients, and for some cases hessians as well as approximations thereof.
As a user, you can easily define custom loss functions.
For those, you can decide to provide analytical gradients or use finite difference approximation / automatic differentiation.
You can choose to mix and match loss functions natively found in this package and those you provide.
In such cases, you optimize over a sum of different objectives (e.g. ML + Ridge).
This mix and match strategy also applies to gradients, where you may supply analytic gradients or opt for automatic differentiation or mix analytical and automatic differentiation.

# You may consider using this package if:

- you want to extend SEM (e.g. add a new objective function) and need an extensible framework
- you want to extend SEM, and your implementation needs to be fast (because you want to do a simulation, for example)
- you want to fit the same model(s) to many datasets (bootstrapping, simulation studies)
- you are planning a study and would like to do power simulations

The package makes use of
- Symbolics.jl for symbolically precomputing parts of the objective and gradients to generate fast, specialized functions.
- SparseArrays.jl to speed up symbolic computations.
- Optim.jl and NLopt.jl to provide a range of different Optimizers/Linesearches.
- FiniteDiff.jl and ForwardDiff.jl to provide gradients for user-defined loss functions.

# At the moment, we are still working on:
- optimizing performance for big models (with hundreds of parameters)

# Questions?

If you have questions you may ask them here in the [issues](https://github.com/StructuralEquationModels/StructuralEquationModels.jl/issues/new).
Please observe our [code of conduct](/CODE_OF_CONDUCT.md).
