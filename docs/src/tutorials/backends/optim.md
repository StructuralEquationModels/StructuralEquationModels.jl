# Using Optim.jl

[`SemOptimizerOptim`](@ref) implements the connection to `Optim.jl`.
It takes two arguments, `algorithm` and `options`.
The defaults are LBFGS as the optimization algorithm and the standard options from `Optim.jl`.
We can load the `Optim` and `LineSearches` packages to choose something different:

```julia
using Optim, LineSearches

my_optimizer = SemOptimizerOptim(
    algorithm = Newton(
        linesearch = BackTracking(order=3), 
        alphaguess = InitialHagerZhang()
        ),
    options = Optim.Options(show_trace = true) 
    )
```

A model with this optimizer object will use Newtons method (= hessian based optimization) with a back tracking linesearch and a certain initial step length guess. Also, the trace of the optimization will be printed to the console.

For a list of all available algorithms and options, we refer to [this page](https://julianlsolvers.github.io/Optim.jl/stable/#user/config/) of the `Optim.jl` manual.