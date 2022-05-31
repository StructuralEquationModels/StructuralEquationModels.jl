# Model construction

There are two different ways of specifying a SEM in our package. You can use the [Outer Constructor](@ref) oder [Build by parts](@ref). 
All tutorials until now used the outer constructor `Sem(specification = ..., data = ..., ...)`, which is normally the more convenient way.
However, our package is build for extensibility, so there may be cases where **user-defined** parts of a model do not work with the outer constructor.
Therefore, building the model by parts is always available as a fallback.

## What is a Structural Equation Model?

In our package, every Structural Equation Model (`Sem`) consists of four parts:
- image of Sem here -

Every part has different subtypes that can serve in the respective place. Those parts are interchangable like Legos (we will make a few examples at the end).

The difference in both contruction methods (building by parts or with the outer constructor) is only about how to arrive at the final model; the choice about which 'Legos' to put together is independet from it. However, the outer constructor simply has some default values that are put in place unless you demand something else.

The rest of this page is about which 'Legos' are available for each part. The specific pages [Outer Constructor](@ref) and [Build by parts](@ref) are about how to stick them together to a final model.

## observed

The `observed` part contains all necessary information about the observed data. Currently, we have three options: [`SemObsData`](@ref) for fully observed datasets, [`SemObsCovariance`](@ref) for observed covariances (and means) and [`SemObsMissing`](@ref) for data that contains missing values.

## imply
The imply part is what your model implies about the data, for example, the model-implied covariance matrix. 
There are two options at the moment: `RAM`, which uses the RAM to compute the model implied covariance matrix, and `RAMSymbolic` which only differes from the `RAM` type in that it symbolically pre-computes part of the model, which increases subsequent performance in model fitting (see [Symbolic precomputation](@ref)).

## loss
The loss part specifies the objective function (or loss function) that is optimized to find the parameter estimates.
If it contains more then one loss function, we find the parameters by minimizing the sum of loss functions (for example in maximum likelihood estimation + ridge regularization).
Available loss functions are
- SemML: maximum likelihood estimation
- SemWLS: weighted leat squares estimation
- SemFIML: full-information maximum likelihood estimation
- SemRidge: ridge regularization

## diff
The diff part of a model connects to the numerical optimization backend used to fit the model. It can be used to control options like the optimization algorithm, linesearch, stopping criteria, etc. There are currently two available backends, `SemDiffOptim` connecting to the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) backend, and `SemDiffNLopt` connecting to the [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) backend.

## API
```@docs
SemObsData
SemObsCovariance
SemObsMissing
SemML
SemFIML
SemRidge
RAM
ParameterTable
EnsembleParameterTable
```