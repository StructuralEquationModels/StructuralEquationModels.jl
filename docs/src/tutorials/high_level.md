# Using the high-level interface

We will fit the following example model:

-- include image here --

We can use the `StenoGraph` package to define our model, which has a similar syntax to the R-package `lavaan`:

```julia
observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3]
latent_vars = [:ξ, :η]

graph = @StenoGraph begin
    # loadings and regressions
    [fixed(1)*x1, x2, x3] ← ξ → η → [fixed(1)*y1, y2, y3]
    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)
end
```

We then use this graph to define a `ParameterTable` object:

```julia
partable = ParameterTable(
    latent_vars = latent_vars, 
    observed_vars = observed_vars, 
    graph = graph)
```

We will use the example data from our package

-- load example data --

And specify our model as

```julia
model = Sem(
    specification = partable,
    data = dat
)
```

We can now fit the model via

```julia
model_fit = sem_fit(model)
```

and compute fit measures and standard errors via

```julia
fitmeasures(model_fit)
se(model_fit)
```

we can also update the parameter table 

-- update partable --