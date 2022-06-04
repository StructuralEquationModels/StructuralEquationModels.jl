# Starting values

The `sem_fit` function has a keyword argument that takes either a vector of starting values or a function that takes a model as input to compute starting values. Current options are `start_fabin3` for fabin 3 starting values (XXX) or `start_simple` for simple starting values. Additional keyword arguments to `sem_fit` are passed to the starting value function. For example,

```julia
    sem_fit(
        model; 
        start_val = start_simple,
        start_covariances_latent = 0.5
    )
```

uses simple starting values with `0.5` as a starting value for covariances between latent variables.