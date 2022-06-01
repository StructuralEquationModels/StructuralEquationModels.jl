# Our Concept of a Structural Equation Model

In our package, every Structural Equation Model (`Sem`) consists of four parts:

![SEM concept](../assets/concept.svg)

Those parts are interchangable building blocks (like 'Legos'), i.e. there are different pieces available you can choose in the `SemObs` slot of the model, and stick them together with other pieces that can serve in the `SemImply` part.

For example, to build a model for maximum likelihood estimation with the NLopt optimization suite as a backend you would choose `SemML` as a `SemLossFunction` and `SemDiffNLopt` as the `SemDiff` part.

The rest of this page is about which 'Legos' are available for each part.
The section [Model construction](@ref) is about how to stick them together to a final model.

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