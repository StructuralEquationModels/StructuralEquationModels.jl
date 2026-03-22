# Type hierarchy

The type hierarchy is implemented in `"src/types.jl"`.

[`AbstractLoss`](@ref): is the base abstract type for all loss functions:
- `SemLoss{O <: SemObserved, I <: SemImplied}`: is the subtype of `AbstractLoss`, which is the
  base for all SEM-specific loss functions ([`SemML`](@ref), [`SemWLS`](@ref) etc) that
  evaluate how closely the implied covariation structure (represented by the object of type `I`)
  matches the observed one (contained in the object of type `O`);
- regularizing terms (e.g. [`SemRidge`](@ref)) are implemented as subtypes of `AbstractLoss`.

[`AbstractSem`](@ref) is the base abstract type for all SEM models. It has two concrete subtypes:
- `Sem{L <: Tuple} <: AbstractSem`: the main SEM model type that implements a list of weighted
loss terms (using [`LossTerm`](@ref) wrapper around `AbstractLoss`) and allows modeling both single
and multi-group SEMs and combining them with regularization terms.
- `SemFiniteDiff{S <: AbstractSem} <: AbstractSem`: a wrapper around any `AbstractSem` that
  substitutes dedicated gradient/hessian evaluation with finite difference approximation.
