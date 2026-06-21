# Model inspection

```@setup colored
using StructuralEquationModels

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin

    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8

    # latent regressions
    ind60 → dem60
    dem60 → dem65
    ind60 → dem65

    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6

end

partable = ParameterTable(
    graph,
    latent_vars = latent_vars,
    observed_vars = observed_vars)

data = example_data("political_democracy")

model = Sem(
    specification = partable,
    data = data
)

model_fit = fit(model)
```

After you fitted a model,

```julia
model_fit = fit(model)
```

you end up with an object of type [`SemFit`](@ref).

You can get some more information about it by using the `details` function:

```@example colored; ansicolor = true
details(model_fit)
```

To compute fit measures, we use

```@example colored; ansicolor = true
fit_measures(model_fit)
```

or compute them individually:

```@example colored; ansicolor = true
AIC(model_fit)
```

A list of available [Fit measures](@ref) is at the end of this page.

To inspect the parameter estimates, we can update a `ParameterTable` object and call `details` on it:

```@example colored; ansicolor = true; output = false
update_estimate!(partable, model_fit)

details(partable)
```

We can also update the `ParameterTable` object with other information via [`update_partable!`](@ref). For example, we can obtain standard errors from the inverse Hessian with [`se_hessian`](@ref) or by bootstrapping with [`se_bootstrap`](@ref), and add both to the table to compare them:

```@example colored; ansicolor = true
se_bs = se_bootstrap(model_fit; n_boot = 20)
se_he = se_hessian(model_fit)

update_partable!(partable, :se_hessian, model_fit, se_he)
update_partable!(partable, :se_bootstrap, model_fit, se_bs)

details(partable)
```

From a vector of standard errors we can also compute *p*-values and confidence intervals for the parameter estimates.
[`z_test!`](@ref) adds the two-sided *p*-value of the test that each parameter is zero (using `z = estimate / se`), and [`normal_CI!`](@ref) adds the lower and upper bounds of a normal-theory confidence interval (95% by default).
Both update the `ParameterTable` in place:

```@example colored; ansicolor = true
z_test!(partable, model_fit, se_he)
normal_CI!(partable, model_fit, se_he)

details(partable; show_columns = [:to, :estimate, :p_value, :ci_lower, :ci_upper])
```

The non-mutating variants [`z_test`](@ref) and [`normal_CI`](@ref) return the values instead of writing them to the table.

Beyond standard errors, [`bootstrap`](@ref) draws bootstrap samples of an arbitrary statistic of a fitted model (not only the parameter estimates), while [`se_bootstrap`](@ref) is a convenience wrapper returning bootstrapped standard errors.
Both support parallel resampling across the available Julia threads via the `parallel = true` keyword.

## Export results

You may convert a `ParameterTable` to a `DataFrame` and use the [`DataFrames`](https://github.com/JuliaData/DataFrames.jl) package for further analysis (or to save it to your hard drive).

```@example colored; ansicolor = true
using DataFrames

parameters_df = DataFrame(partable);
```

# API - model inspection

```@docs
details
update_estimate!
update_partable!
```

## Additional functions

Additional functions that can be used to extract information from a `SemFit` object:

```@docs
SemFit
params
param_labels
nparams
nsamples
nobserved_vars
```

## Fit measures

```@docs
fit_measures
AIC
BIC
χ²
dof
minus2ll
p_value
RMSEA
CFI
```

## Standard errors and inference

```@docs
se_hessian
se_bootstrap
bootstrap
StructuralEquationModels.BootstrapResult
z_test
z_test!
normal_CI
normal_CI!
```
