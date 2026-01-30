# Using Optim.jl

[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) is the default optimization engine of *SEM.jl*,
see ... for a full list of its parameters.
It defaults to the LBFGS optimization, but we can load the `Optim` and `LineSearches` packages
and specify BFGS (!not L-BFGS) with a back-tracking linesearch and Hager-Zhang initial step length guess:

```julia
using Optim, LineSearches

my_optimizer = SemOptimizer(
    algorithm = BFGS(
        linesearch = BackTracking(order=3),
        alphaguess = InitialHagerZhang()
    ),
    options = Optim.Options(show_trace = true)
)
```

Note that we used `options` to print the optimization progress to the console.

To see how to use the optimizer to actually fit a model now, check out the [Model fitting](@ref) section.

For a list of all available algorithms and options, we refer to [this page](https://julianlsolvers.github.io/Optim.jl/stable/#user/config/) of the `Optim.jl` manual.