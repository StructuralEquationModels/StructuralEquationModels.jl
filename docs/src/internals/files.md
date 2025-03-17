# Files

We briefly describe the file and folder structure of the package.

## Source code

Source code is in the `"src"` folder:

`"src"`
- `"StructuralEquationModels.jl"` defines the module and the exported objects
- `"types.jl"` defines all abstract types and the basic type hierarchy
- `"objective_gradient_hessian.jl"` contains methods for computing objective, gradient and hessian values for different model types as well as generic fallback methods
- The four folders `"observed"`, `"implied"`, `"loss"` and `"diff"` contain implementations of specific subtypes (for example, the `"loss"` folder contains a file `"ML.jl"` that implements the `SemML` loss function).
- `"optimizer"` contains connections to different optimization backends (aka methods for `fit`)
    - `"optim.jl"`: connection to the `Optim.jl` package
- `"frontend"` contains user-facing functions
    - `"specification"` contains functionality for model specification
    - `"fit"` contains functionality for model assessment, like fit measures and standard errors
- `"additional_functions"` contains helper functions for simulations, loading artifacts (example data) and various other things

Code for the package extentions can be found in the `"ext"` folder:
- `"SEMNLOptExt"` for connection to `NLopt.jl`.
- `"SEMProximalOptExt"` for connection to `ProximalAlgorithms.jl`.

## Tests and Documentation

Tests are in the `"test"` folder, documentation in the `"docs"` folder.