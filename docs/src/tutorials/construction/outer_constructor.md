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