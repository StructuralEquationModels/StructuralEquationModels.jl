# A first model

In this tutorial, we will fit our very first Structural Equation Model with our package. 
The example we are using is from [the `lavaan` tutorial](https://lavaan.ugent.be/tutorial/sem.html), so it may be familiar.
It looks like this:

![Visualization of Political Democracy model](https://lavaan.ugent.be/tutorial/figure/sem-1.png)

[2022 © Yves Roseel](https://lavaan.ugent.be)

We assume the `StructuralEquationModels` package is already installed. To use it in the current session, we run

```@example high_level
using StructuralEquationModels
```

We then first define the graph of our model in a syntax which is similar to the R-package `lavaan`:

```@setup high_level
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
```

```julia
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
```

We then use this graph to define a `ParameterTable` object

```@example high_level; ansicolor = true
partable = ParameterTable(
    latent_vars = latent_vars, 
    observed_vars = observed_vars, 
    graph = graph)
```

load the example data

```julia
data = example_data("political_democracy")
```

```@setup high_level
data = example_data("political_democracy")
```

and specify our model as

```@example high_level; ansicolor = true
model = Sem(
    specification = partable,
    data = data
)
```

We can now fit the model via

```@example high_level; ansicolor = true
model_fit = sem_fit(model)
```

and compute fit measures as

```@example high_level; ansicolor = true
fit_measures(model_fit)
```

We can also get a bit more information about the fitted model via the `sem_summary()` function:

```@example high_level; ansicolor = true
sem_summary(model_fit)
```

To investigate the parameter estimates, we can update our `partable` object to contain the new estimates:

```@example high_level; ansicolor = true
update_estimate!(partable, model_fit)
```

and investigate the solution with

```@example high_level; ansicolor = true
sem_summary(partable)
```

Congratulations, you fitted and inspected your very first model! To learn more about the different parts, 
you may visit the sections on [Model specification](@ref), [Model construction](@ref), [Model fitting](@ref) and
[Model inspection](@ref).

If you want to learn how to extend the package (e.g., add a new loss function), you may visit the developer documentation (XXX).
