# Model inspection

```@setup colored
using StructuralEquationModels 

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin

    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8

    # latent regressions
    ind60 → dem60
    dem60 → dem65
    ind60 → dem65

    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6

end

partable = ParameterTable(
    latent_vars = latent_vars, 
    observed_vars = observed_vars, 
    graph = graph)

data = example_data("political_democracy")

model = Sem(
    specification = partable,
    data = data
)

model_fit = sem_fit(model)
```

After you fitted a model,

```julia
model_fit = sem_fit(model)
```

you and up with an object of type `SemFit`.

You can get some more information about it by using the `sem_summary` function:

```@example colored; ansicolor = true
sem_summary(model_fit)
```

To compute fit measures, we use

```@example colored; ansicolor = true
fit_measures(model_fit)
```

To inspect the parameter estimates, we can update a `ParameterTable` object and call `sem_summary` on it:

```@example colored; ansicolor = true; output = false
update_estimate!(partable, model_fit)

sem_summary(partable)
```

We can also update the `ParameterTable` object with other information. For example, if we want to compare hessian-based and bootstrap-based standard errors, we may write

```@example color; ansicolor = true

```