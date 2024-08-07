# Multigroup models

```@setup mg
using StructuralEquationModels
```

As an example, we will fit the model from [the `lavaan` tutorial](https://lavaan.ugent.be/tutorial/groups.html) with loadings constrained to equality across groups.

We first load the example data and split it between groups:

```@setup mg
dat = example_data("holzinger_swineford")

dat_g1 = dat[dat.school .== "Pasteur", :]
dat_g2 = dat[dat.school .== "Grant-White", :]
```

```julia
dat = example_data("holzinger_swineford")

dat_g1 = dat[dat.school .== "Pasteur", :]
dat_g2 = dat[dat.school .== "Grant-White", :]
```

We then specify our model via the graph interface:

```@setup mg
latent_vars = [:visual, :textual, :speed]
observed_vars = Symbol.(:x, 1:9)

graph = @StenoGraph begin
    # measurement model
    visual  → fixed(1.0, 1.0)*x1 + label(:λ₂, :λ₂)*x2 + label(:λ₃, :λ₃)*x3
    textual → fixed(1.0, 1.0)*x4 + label(:λ₅, :λ₅)*x5 + label(:λ₆, :λ₆)*x6
    speed   → fixed(1.0, 1.0)*x7 + label(:λ₈, :λ₈)*x8 + label(:λ₉, :λ₉)*x9
    # variances and covariances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars)   ⇔ _(latent_vars)
end
```

```julia
latent_vars = [:visual, :textual, :speed]
observed_vars = Symbol.(:x, 1:9)

graph = @StenoGraph begin
    # measurement model
    visual  → fixed(1, 1)*x1 + label(:λ₂, :λ₂)*x2 + label(:λ₃, :λ₃)*x3
    textual → fixed(1, 1)*x4 + label(:λ₅, :λ₅)*x5 + label(:λ₆, :λ₆)*x6
    speed   → fixed(1, 1)*x7 + label(:λ₈, :λ₈)*x8 + label(:λ₉, :λ₉)*x9
    # variances and covariances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars)   ⇔ _(latent_vars)
end
```

You can pass multiple arguments to `fix()` and `label()` for each group. Parameters with the same label (within and across groups) are constrained to be equal. To fix a parameter in one group, but estimate it freely in the other, you may write `fix(NaN, 4.3)`.

You can then use the resulting graph to specify an `EnsembleParameterTable`

```@example mg; ansicolor = true
groups = [:Pasteur, :Grant_White]

partable = EnsembleParameterTable(
    graph, 
    observed_vars = observed_vars,
    latent_vars = latent_vars,
    groups = groups)
```

The parameter table can be used to create a `Dict` of RAMMatrices with keys equal to the group names and parameter tables as values:

```@example mg; ansicolor = true
specification = convert(Dict{Symbol, RAMMatrices}, partable)
```

That is, you can asses the group-specific `RAMMatrices` as `specification[:group_name]`.

!!! note "A different way to specify"
    Instead of choosing the workflow "Graph -> EnsembleParameterTable -> RAMMatrices", you may also directly specify RAMMatrices for each group (for an example see [this test](https://github.com/StructuralEquationModels/StructuralEquationModels.jl/blob/main/test/examples/multigroup/multigroup.jl)).

The next step is to construct the model:

```@example mg; ansicolor = true
model_g1 = Sem(
    specification = specification[:Pasteur],
    data = dat_g1
)

model_g2 = Sem(
    specification = specification[:Grant_White],
    data = dat_g2
)

model_ml_multigroup = SemEnsemble(model_g1, model_g2)
```

We now fit the model and inspect the parameter estimates:

```@example mg; ansicolor = true
solution = sem_fit(model_ml_multigroup)
update_estimate!(partable, solution)
sem_summary(partable)
```

Other things you can query about your fitted model (fit measures, standard errors, etc.) are described in the section [Model inspection](@ref) and work the same way for multigroup models.