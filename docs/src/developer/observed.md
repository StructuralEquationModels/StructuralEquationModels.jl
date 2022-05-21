# Custom observed types

The implementation of new observed types is very similar to loss functions, so we will just go over it briefly (for additional information, revisit [Custom loss functions](@ref)).

First, we need to define a new struct that is a subtype of `SemObs`:

```julia
struct MyObserved <: SemObs
    ...
end
```

Additionally, we can write an outer constructor that will typically depend on the keyword argument `data = ...`:

```julia
function MyObserved(;data, kwargs...)
    ...
    return MyObserved(...)
end
```

To compute some fit indices, you need to provide methods for

```julia
# Number of observed datapoints
n_obs(observed::MyObserved) = ...
# Number of manifest variables
n_man(observed::MyObserved) = ...
```

As always, you can add additional methods for properties that imply types and loss function want to access, for example (from the `SemObsCommon` implementation):

```julia
obs_cov(observed::SemObsCommon) = observed.obs_cov
```