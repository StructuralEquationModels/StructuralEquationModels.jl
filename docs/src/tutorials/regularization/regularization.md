# Regularization

## Setup

For ridge regularization, you can simply use `SemRidge` as an additional loss function
(for example, a model with the loss functions `SemML` and `SemRidge` corresponds to ridge-regularized maximum likelihood estimation).

You can define lasso, elastic net and other forms of regularization using [`ProximalOperators.jl`](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
and optimize the SEM model with [`ProximalAlgorithms.jl`](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)
that provides so-called *proximal optimization* algorithms.

```@setup reg
using StructuralEquationModels, ProximalAlgorithms, ProximalOperators
```

```julia
using Pkg
Pkg.add("ProximalAlgorithms")
Pkg.add("ProximalOperators")

using StructuralEquationModels, ProximalAlgorithms, ProximalOperators
```

## Proximal optimization

With *ProximalAlgorithms* package loaded, it is now possible to use `:Proximal` optimization engine
in `SemOptimizer` for estimating regularized models.

```julia
SemOptimizer(;
    engine = :Proximal,
    algorithm = ProximalAlgorithms.PANOC(),
    operator_g,
    operator_h = nothing
)
```

The *proximal operator* (aka the *regularization function*) is passed as `operator_g`, see [available operators](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/).
The `algorithm` is chosen from one of the [available algorithms](https://juliafirstorder.github.io/ProximalAlgorithms.jl/stable/guide/implemented_algorithms/).

## First example - lasso

To show how it works, let's revisit [A first model](@ref):

```@example reg
observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin

    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8

    dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60

    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    y1 ↔ label(:cov_15)*y5
    y2 ↔ label(:cov_24)*y4 + label(:cov_26)*y6
    y3 ↔ label(:cov_37)*y7
    y4 ↔ label(:cov_48)*y8
    y6 ↔ label(:cov_68)*y8

end

partable = ParameterTable(
    graph,
    latent_vars = latent_vars,
    observed_vars = observed_vars
)

data = example_data("political_democracy")

model = Sem(
    specification = partable,
    data = data
)
```

We labeled the covariances between the items because we want to regularize those:

```@example reg
cov_inds = getindex.(
    Ref(param_indices(model)),
    [:cov_15, :cov_24, :cov_26, :cov_37, :cov_48, :cov_68])
```

In the following, we fit the same model with lasso regularization of those covariances.
The lasso penalty is defined as

```math
\sum \lambda_i \lvert \theta_i \rvert
```

In `ProximalOperators.jl`, lasso regularization is represented by the [`NormL1`](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/#ProximalOperators.NormL1) operator. It allows controlling the amount of
regularization individually for each SEM model parameter via the vector of hyperparameters (`λ`).
To regularize only the observed item covariances, we define `λ` as

```@example reg
λ = zeros(31); λ[cov_inds] .= 0.02

optimizer_lasso = SemOptimizer(
    engine = :Proximal,
    operator_g = NormL1(λ)
)
```

Let's fit the regularized model

```@example reg

fit_lasso = fit(optimizer_lasso, model)
```

and compare the solution to unregularizted estimates:

```@example reg
sem_fit = fit(model)

update_estimate!(partable, sem_fit)

update_partable!(partable, :estimate_lasso, fit_lasso, solution(fit_lasso))

details(partable)
```

Instead of explicitly defining a `SemOptimizer` object, you can also pass `engine = :Proximal`
and additional keyword arguments directly to the `fit` function:

```@example reg
fit_lasso2 = fit(model; engine = :Proximal, operator_g = NormL1(λ))
```

## Second example - mixed l1 and l0 regularization

You can choose to penalize different parameters with different types of regularization functions.
Let's use the *lasso* (*l1*) again on the covariances, but additionally penalize the error variances of the observed items via *l0* regularization.

The *l0* penalty is defined as
```math
l_0 = \lambda \mathrm{nnz}(\theta)
```

Since we apply *l1* and *l0* to the disjoint sets of parameters, this regularization could be represented as
as sum of *separable proximal operators* (i.e. no parameter is penalized twice)
implemented by the [`SlicedSeparableSum`](https://juliafirstorder.github.io/ProximalOperators.jl/stable/calculus/#ProximalOperators.SlicedSeparableSum) operator:

```@example reg
l0_and_l1_reg = SlicedSeparableSum((NormL0(20.0), NormL1(0.02), NormL0(0.0)), ([cov_inds], [9:11], [vcat(1:8, 12:25)]))

fit_mixed = fit(model; engine = :Proximal, operator_g = l0_and_l1_reg)
```

Let's again compare the different results:

```@example reg
update_partable!(partable, :estimate_mixed, fit_mixed, solution(fit_mixed))

details(partable)
```