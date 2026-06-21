# Collections

With *StructuralEquationModels.jl*, you can fit weighted sums of structural equation models.
The most common use case for this are [Multigroup models](@ref).
Another use case may be optimizing the sum of loss functions for some of which you do know the analytic gradient, but not for others.
In this case, [`FiniteDiffWrapper`](@ref StructuralEquationModels.FiniteDiffWrapper) can generate a wrapper around the specific `SemLoss` term. The wrapper loss term will
only use the objective of the original term to calculate its gradient using finite difference approximation.

```julia
loss_1 = SemML(observed_1, implied_1)
loss_2 = SemML(observed_2, implied_2)
loss_2_findiff = FiniteDiffWrapper(loss_2)
```

To construct `Sem` from the the individual `SemLoss` (or other `AbstractLoss`) terms, they are
just passed to the `Sem` constructor:

```julia
model = Sem(loss_1, loss_2)
model_findiff = Sem(loss_1, loss_2_findiff)
```

It is also possible to use finite difference for the entire `Sem` model:

```julia
model_findiff2 = FiniteDiffWrapper(model)
```

The weighting scheme of the SEM loss terms is specified using the `default_sem_weights` argument of the `Sem` constructor.
The available schemes are:
- `:nsamples_corrected` (the default): like `:nsamples`, but applies a loss-type-specific correction
  to the sample counts (e.g. ``N_{term} - 1`` for maximum likelihood and weighted least squares). For FIML the correction is zero, so it coincides with `:nsamples`,
- `:nsamples`: weights each SEM term by ``N_{term}/N_{total}``, i.e. by the (uncorrected) number of
  observations in its data,
- `:uniform`: weights each of the ``k`` SEM terms by ``1/k``,
- `:one`: leaves all SEM terms unweighted.

The weights for the loss terms (both SEM and regularization) can also be explicitly specified using the pair syntax `loss => weight`:

```julia
model_weighted = Sem(loss_1 => 0.5, loss_2 => 1.0)
```

`Sem` supports assigning a unique identifier to each loss term, which is useful for complex multi-term models.
The syntax is `id => loss`, or `id => loss => weight`:

```julia
model2 = Sem(:main => loss_1, :alt => loss_2)
model2_weighted = Sem(:main => loss_1 => 0.5, :alt => loss_2 => 1.0)
```

# API - collections

```@docs
Sem
SemFiniteDiff
AbstractSem
StructuralEquationModels.LossTerm
StructuralEquationModels.FiniteDiffWrapper
```