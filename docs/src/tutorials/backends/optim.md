# Using Optim.jl

[`SemOptimizerOptim`](@ref) implements the connection to `Optim.jl`.
It takes two arguments, `algorithm` and `options`.
The defaults are LBFGS as the optimization algorithm and the standard options from `Optim.jl`.
We can load the `Optim` and `LineSearches` packages to choose something different:

```julia
using Optim, LineSearches

my_optimizer = SemOptimizerOptim(
    algorithm = BFGS(
        linesearch = BackTracking(order=3), 
        alphaguess = InitialHagerZhang()
        ),
    options = Optim.Options(show_trace = true) 
    )
```

This optimizer will use BFGS (!not L-BFGS) with a back tracking linesearch and a certain initial step length guess. Also, the trace of the optimization will be printed to the console.

To see how to use the optimizer to actually fit a model now, check out the [Model fitting](@ref) section.

For a list of all available algorithms and options, we refer to [this page](https://julianlsolvers.github.io/Optim.jl/stable/#user/config/) of the `Optim.jl` manual.