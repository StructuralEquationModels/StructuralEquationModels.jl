# Starting values

The `fit` function has a keyword argument that takes either a vector of starting values or a function that takes a model as input to compute starting values. Current options are `start_fabin3` for fabin 3 starting values [^Hägglund82] or `start_simple` for simple starting values. Additional keyword arguments to `fit` are passed to the starting value function. For example,

```julia
    fit(
        model; 
        start_val = start_simple,
        start_covariances_latent = 0.5
    )
```

uses simple starting values with `0.5` as a starting value for covariances between latent variables.

[^Hägglund82]: Hägglund, G. (1982). Factor analysis by instrumental variables methods. Psychometrika, 47(2), 209-222.