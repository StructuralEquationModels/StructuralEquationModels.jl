# Model specification

We provide different interfaces for specifying a model: the [Graph interface](@ref), the [ParameterTable interface](@ref),
and the [RAMMatrices interface](@ref). These different specification objects can be (and are internally) converted to each other; but not every conversion is possible - see this picture:

- imagine flowchart here -

In general (and especially if you come from `lavaan`), it is the easiest to follow the steps from the page [A first model](@ref), that is specify a graph object, convert it to a prameter table, and use this parameter table to construct your models:

```julia
observed_vars = ...
latent_vars   = ...

graph = @StenoGraph begin
    ...
end

partable = ParameterTable(
    latent_vars = latent_vars, 
    observed_vars = observed_vars, 
    graph = graph)

model = Sem(
    specification = partable,
    ...
)
```

If you have an `OpenMx` background, and are familiar with their way of specifying structural equation models via RAM matrices,
the [RAMMatrices interface](@ref) may be of interest for you.