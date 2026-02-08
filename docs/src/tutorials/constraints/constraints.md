# Constrained optimization

*SEM.jl* allows to fit models with additional constraints imposed on the parameters.

## Using the NLopt engine

*NLopt.jl* is one of *SEM.jl* optimization engines that supports constrained optimization.
In the example below we show how to specify constraints for the *SEM* model when using *NLopt*.

### Define an example model

Let's revisit our model from [A first model](@ref) and fit it first without constraints:

```@example constraints
using StructuralEquationModels

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin

    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + label(:λ₂)*y2 + label(:λ₃)*y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8

    # latent regressions
    ind60 → dem60
    dem60 → dem65
    ind60 → label(:λₗ)*dem65

    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)

    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ label(:y3y7)*y7
    y8 ↔ label(:y8y4)*y4 + y6

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

update_estimate!(partable, model_fit)

details(partable)
```

### Define the constraints

Let's introduce some constraints (they are not based on any real properties of the underlying study and serve only as an example):
1. **Equality constraint**: The covariances `y3 ↔ y7` and `y8 ↔ y4` should sum up to `1`.
2. **Inequality constraint**: The difference between the loadings `dem60 → y2` and `dem60 → y3` should be smaller than `0.1`
3. **Bound constraint**: The directed effect from  `ind60 → dem65` should be smaller than `0.5`

Since *NLopt* does not have access to the SEM parameter names, its constaints are defined on the vector of all SEM parameters.
We have to look up the indices of the parameters involved in the constraints to construct the respective functions.

```@example constraints
parind = param_indices(model)
parind[:y3y7] # 29
```

The bound constraint is easy to specify: just give a vector of upper or lower bounds for each parameter.
In our example, only the parameter labeled `:λₗ` has an upper bound, and the number of total parameters is `n_par(model) = 31`, so

```@example constraints
upper_bounds = fill(Inf, 31)
upper_bounds[parind[:λₗ]] = 0.5
```

The equailty and inequality constraints have to be reformulated in the `f(θ) = 0` or `f(θ) ≤ 0` form,
where `θ` is the vector of SEM parameters:
1. `f(θ) = 0`, where `f(θ) = y3 ↔ y7 + y8 ↔ y4 - 1`
2. `g(θ) ≤ 0`, where `g(θ) = dem60 → y2 - dem60 → y3 - 0.1`

If the optimization algorithm needs gradients, it will pass the `gradient` vector that is of the same size as the parameters,
and the constraint function has to calculate the gradient in-place.

```@example constraints
parind[:y3y7] # 29
parind[:y8y4] # 30
# θ[29] + θ[30] - 1 = 0.0
function f(θ, gradient)
    if length(gradient) > 0
        gradient .= 0.0
        gradient[29] = 1.0
        gradient[30] = 1.0
    end
    return θ[29] + θ[30] - 1
end

parind[:λ₂] # 3
parind[:λ₃] # 4
# θ[3] - θ[4] - 0.1 ≤ 0
function g(θ, gradient)
    if length(gradient) > 0
        gradient .= 0.0
        gradient[3] = 1.0
        gradient[4] = -1.0
    end
    θ[3] - θ[4] - 0.1
end
```

In *NLopt*, vector-valued constraints are also possible, but we refer to the documentation for that.

### Fit the model

Now we can construct the *SemOptimizer* that will use the *NLopt* engine for constrained optimization.

```@example constraints
using NLopt

constrained_optimizer = SemOptimizer(
    engine = :NLopt,
    algorithm = :AUGLAG,
    options = Dict(:upper_bounds => upper_bounds, :xtol_abs => 1e-4),
    local_algorithm = :LD_LBFGS,
    equality_constraints = (f => 1e-8),
    inequality_constraints = (g => 1e-8),
)
```

As you see, the equality and inequality constraints are passed as keyword arguments, and the bounds are passed as options for the (outer) optimization algorithm.
Additionally, for equality and inequality constraints, a feasibility tolerance can be specified that controls if a solution can be accepted, even if it violates the constraints by a small amount.
Especially for equality constraints, it is recommended to allow for a small positive tolerance.
In this example, we set both tolerances to `1e-8`.

!!! warning "Convergence criteria"
    We have often observed that the default convergence criteria in NLopt lead to non-convergence flags.
    Indeed, this example does not convergence with default criteria.
    As you see above, we used a relatively liberal absolute tolerance in the optimization parameters of 1e-4.
    This should not be a problem in most cases, as the sampling variance in (almost all) structural equation models
    should lead to uncertainty in the parameter estimates that are orders of magnitude larger.
    We nontheless recommend choosing a convergence criterion with care (i.e. w.r.t. the scale of your parameters),
    inspecting the solutions for plausibility, and comparing them to unconstrained solutions.

We now have everything to fit our model under constraints:

```@example constraints
model_fit_constrained = fit(constrained_optimizer, model)
```

As you can see, the optimizer converged (`:XTOL_REACHED`) and investigating the solution yields

```@example constraints
update_partable!(
    partable,
    :estimate_constr,
    model_fit_constrained,
    solution(model_fit_constrained),
)

details(partable)
```

As we can see, the constrained solution is very close to the original solution (compare the columns estimate and estimate_constr), with the difference that the constrained parameters fulfill their constraints.
As all parameters are estimated simultaneously, it is expexted that some unconstrained parameters are also affected (e.g., the constraint on `dem60 → y2` leads to a higher estimate of the residual variance `y2 ↔ y2`).

## Using the Optim.jl backend

Information about constrained optimization using `Optim.jl` can be found in the packages [documentation](https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/).