# Constrained optimization

## Using the NLopt backend

!!! warning "Experimental feature"
    Altough we found that constrained optimization via the NLopt backend seems to work well, the convergence flags do not work properly - that is, we often get convergence flags indicating non-convergence even if the model converged. We are currently trying to find a solution for this.

### Define an example model

Let's revisit our model from [A first model](@ref):

```@example constraints
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
    latent_vars = latent_vars, 
    observed_vars = observed_vars, 
    graph = graph)

data = example_data("political_democracy")

model = Sem(
    specification = partable,
    data = data
)

model_fit = sem_fit(model)

update_estimate!(partable, model_fit)

sem_summary(partable)
```

### Define the constraints

Let's introduce some constraints:
1. **Equality constraint**: The covariances `y3 ↔ y7` and `y8 ↔ y4` should sum up to `1`.
2. **Inequality constraint**: The difference between the loadings `dem60 → y2` and `dem60 → y3` should be smaller than `0.1`
3. **Bound constraint**: The directed effect from  `ind60 → dem65` should be smaller than `0.5`

(Of course those constaints only serve an illustratory purpose.)

We first need to get the indices of the respective parameters that are invoved in the constraints. We can look up their labels in the output above, and retrieve their indices as

```@example constraints
parameter_indices = get_identifier_indices([:θ_29, :θ_30, :θ_3, :θ_4, :θ_11], model)
```

The bound constraint is easy to specify: Just give a vector of upper or lower bounds that contains the bound for each parameter. In our example, only parameter number 11 has an upper bound, and the number of total parameters is `n_par(model) = 31`, so we define

```@example constraints
upper_bounds = fill(Inf, 31)
upper_bounds[11] = 0.5
```

The equailty and inequality constraints have to be reformulated to be of the form `x = 0` or `x ≤ 0`:
1. `y3 ↔ y7 + y8 ↔ y4 - 1 = 0`
2. `dem60 → y2 - dem60 → y3 - 0.1 ≤ 0`

Now they can be defined as functions of the parameter vector:

```@example constraints
# θ[29] + θ[30] - 1 = 0.0
function eq_constraint(θ, gradient)
    if length(gradient) > 0
        gradient .= 0.0
        gradient[1] = 1.0
        gradient[2] = 1.0
    end
    return θ[29] + θ[30] - 1
end

# θ[3] - θ[4] - 0.1 ≤ 0
function ineq_constraint(θ, gradient)
    if length(gradient) > 0
        gradient .= 0.0
        gradient[3] = 1.0
        gradient[4] = -1.0
    end
    θ[3] - θ[4] - 0.1
end
```

Here, gradient is a pre-allocated vector for the gradients of the constraint w.r.t. the parameters.

### Fit the model

We now have everything toether to specify and fit our model. First, we specify our optimizer backend as

```@example constraints
constrained_optimizer = SemOptimizerNLopt(
    algorithm = :AUGLAG,
    options = Dict(:upper_bounds => upper_bounds),
    local_algorithm = :LD_LBFGS,
    local_options = Dict(:ftol_rel => 1e-6),
    equality_constraints = NLoptConstraint(;f = eq_constraint, tol = 0.0),
    inequality_constraints = NLoptConstraint(;f = ineq_constraint, tol = 0.0),
)
```

As you see, the equality constraints and inequality constraints are passed as keword arguments, and the bound are passed as options for the (outer) optimization algorithm.

```@example constraints
model_constrained = Sem(
    specification = partable,
    data = data,
    diff = constrained_optimizer
)

model_fit_constrained = sem_fit(model_constrained)
```

As you can see, the convergence flag is `:FAILURE`, but investigating the solution yields

```@example constraints
update_partable!(partable, model_fit_constrained, solution(model_fit_constrained), :estimate_constr)

sem_summary(partable)
```

## Using the Optim.jl backend

Information about constrained optimization using `Optim.jl` can be found in the packages [documentation](https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/).