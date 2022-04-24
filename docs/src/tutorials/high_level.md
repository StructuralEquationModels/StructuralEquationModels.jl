# A first model

In this tutorial, we will fit our very first Structural Equation Model with our package. 
The example we are using is from [the `lavaan` tutorial](https://lavaan.ugent.be/tutorial/sem.html), so it may be familiar.
It looks like this:

-- include image here --

We assume the `StructuralEquationModels` package is already installed. To use it in the current session, we run

```jldoctest high_level
using StructuralEquationModels
```

We then first define the graph of our model in a syntax which is similar to the R-package `lavaan`:

```jldoctest high_level

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

```jldoctest high_level
partable = ParameterTable(
    latent_vars = latent_vars, 
    observed_vars = observed_vars, 
    graph = graph)
```

load the example data

```jldoctest high_level
data = example_data("political_democracy")
```

and specify our model as

```jldoctest high_level
model = Sem(
    specification = partable,
    data = data
)
```

We can now fit the model via

```jldoctest high_level
model_fit = sem_fit(model)
```

and compute fit measures as

```jldoctest high_level
fit_measures(model_fit)
```

We can also get a bit more information about the fitted model via the `sem_summary()` function:

```jldoctest high_level
sem_summary(model_fit)
```

To investigate the parameter estimates, we can update our `partable` object to contain the new estimates:

```jldoctest high_level
update_estimate!(partable, model_fit)
sem_summary(partable)
```

Congratulations, you fitted and inspected your very first model! To learn more about the different parts, 
you may visit the sections on model specification (XXX), model construction (XXX), model fitting (XXX) and
model inspection (XXX).

If you want to learn how to extend the package (e.g., add a new loss function), you may visit the developer documentation (XXX).