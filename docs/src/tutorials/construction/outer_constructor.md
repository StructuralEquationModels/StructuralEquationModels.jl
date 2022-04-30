# Outer Constructor

We already have seen the outer constructor in action in [A first model](@ref):

```julia
model = Sem(
    specification = partable,
    data = data
)

# output

Structural Equation Model
- Loss Functions
   SemML
- Fields
   observed:  SemObsCommon
   imply:     RAM
   diff:      SemDiffOptim
```

The output of this call tells you exactly what model you just constructed (i.e. what the loss functions, observed, imply and diff parts are).

As you can see, by default, we use maximum likelihood estimation, the RAM imply type and the `Optim.jl` optimization backend. 
To choose something different, you can provide it as a keyword argument:

```julia
model = Sem(
    specification = partable,
    data = data,
    observed = ...,
    imply = ...,
    loss = ...,
    diff = ...
)
```

For example, to construct a model for weighted least squares estimation that uses symbolic precomputation and the NLopt backend, write

```julia
model = Sem(
    specification = partable,
    data = data,
    imply = RAMSymbolic,
    loss = SemWLS,
    diff = SemDiffNLopt
)
```

In the section on [Model construction](@ref), we go over the different options you have for each part of the model.
Let's make another example: to use full information maximum likelihood information (FIML), write

```julia
model = Sem(
    specification = partable,
    data = data,
    loss = SemFIML,
    observed = SemObsMissing
)
```

You may also provide addition arguments for specific parts of the model. For example, WLS estimation uses per default

```math
W = \frac{1}{2} D^T(S^{-1}\otimes S^{-1})D
```
as the weight matrix, where D is the so-called duplication matrix and S is the observed covariance matrix. However, you can pass any other weight matrix you want (to achieve UWL, DWLS, ADF estimation, for example) as a keyword argument:

```julia
W = ...

model = Sem(
    specification = partable,
    data = data,
    imply = RAMSymbolic,
    loss = SemWLS
    wls_weight_matrix = W
)

```

To see what additional keyword arguments are supported, you can consult the documentation of the specific part of the model (by either using `help(...)`, or typing `?` in the REPL to enter the help mode and then typing the name of the thing you want to know something about):

```julia
help(SemWLS)

# output

OUTPUT MISSING!

```

## Optimize loss functions without implemented analytic gradient

For loss functions without analytic gradients, it is possible to use finite difference approximation or forward mode automatic differentiation. 
All loss functions provided in the package do have analytic gradients (and some even hessians or approximations thereof), so there is no need do use this feature if you are only working with them.
However, if you implement your own loss function, you do not have to provide analytic gradients.
In that case, you may construct your model just as before, but swap the `Sem` constructor for either `SemFiniteDiff` or `SemForwardDiff`. For example

```julia
model = SemFiniteDiff(
    specification = partable,
    data = data
)
```

constructs a model that will use finite difference approximation if you estimate the parameters via `sem_fit(model)`.
Both `SemFiniteDiff` and `SemForwardDiff` have an additional keyword argument, `has_gradient = ...` that can be set to `true` to indicate that the model has analytic gradients, and only the hessian should be computed via finite difference approximation / automatic differentiation.
For example

```julia
using Optim, LineSearches

model = SemFiniteDiff(
    specification = partable,
    data = data,
    has_gradient = true,
    algorithm = Newton()
)
```

will construct a model that, when fitted, will use [Newton's Method](https://julianlsolvers.github.io/Optim.jl/stable/#algo/newton/) from the `Optim.jl` package with analytic gradients and hessians computed via finite difference approximation.