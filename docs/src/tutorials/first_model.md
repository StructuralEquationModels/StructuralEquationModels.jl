# A first model

In this tutorial, we will fit our very first Structural Equation Model with our package. 
The example we are using is from [the `lavaan` tutorial](https://lavaan.ugent.be/tutorial/sem.html), so it may be familiar.
It looks like this:

-- include image here --

We assume the `StructuralEquationModels` package is already installed. To use it in the current session, we run

```jldoctest high_level; output = false
using StructuralEquationModels

# output

```

We then first define the graph of our model in a syntax which is similar to the R-package `lavaan`:

```jldoctest high_level; output = false
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

# output

ind60 → x1 * Fixed{Tuple{Int64}}((1,))
ind60 → x2
ind60 → x3
dem60 → y1 * Fixed{Tuple{Int64}}((1,))
dem60 → y2
dem60 → y3
dem60 → y4
dem65 → y5 * Fixed{Tuple{Int64}}((1,))
dem65 → y6
dem65 → y7
dem65 → y8
ind60 → dem60
dem60 → dem65
ind60 → dem65
x1 ↔ x1
x2 ↔ x2
x3 ↔ x3
y1 ↔ y1
y2 ↔ y2
y3 ↔ y3
y4 ↔ y4
y5 ↔ y5
y6 ↔ y6
y7 ↔ y7
y8 ↔ y8
ind60 ↔ ind60
dem60 ↔ dem60
dem65 ↔ dem65
y1 ↔ y5
y2 ↔ y4
y2 ↔ y6
y3 ↔ y7
y8 ↔ y4
y8 ↔ y6
```

We then use this graph to define a `ParameterTable` object

```jldoctest high_level; output = false
partable = ParameterTable(
    latent_vars = latent_vars, 
    observed_vars = observed_vars, 
    graph = graph)

# output

 -------- ---------------- -------- ------- ------------- --------- ------------
    from   parameter_type       to    free   value_fixed     start   estimate  ⋯
  Symbol           Symbol   Symbol    Bool       Float64   Float64    Float64  ⋯
 -------- ---------------- -------- ------- ------------- --------- ------------
   ind60                →       x1   false           1.0       0.0        0.0  ⋯
   ind60                →       x2    true           0.0       0.0        0.0  ⋯
   ind60                →       x3    true           0.0       0.0        0.0  ⋯
   dem60                →       y1   false           1.0       0.0        0.0  ⋯
   dem60                →       y2    true           0.0       0.0        0.0  ⋯
   dem60                →       y3    true           0.0       0.0        0.0  ⋯
   dem60                →       y4    true           0.0       0.0        0.0  ⋯
   dem65                →       y5   false           1.0       0.0        0.0  ⋯
   dem65                →       y6    true           0.0       0.0        0.0  ⋯
   dem65                →       y7    true           0.0       0.0        0.0  ⋯
   dem65                →       y8    true           0.0       0.0        0.0  ⋯
   ind60                →    dem60    true           0.0       0.0        0.0  ⋯
   dem60                →    dem65    true           0.0       0.0        0.0  ⋯
   ind60                →    dem65    true           0.0       0.0        0.0  ⋯
      x1                ↔       x1    true           0.0       0.0        0.0  ⋯
    ⋮            ⋮            ⋮        ⋮          ⋮           ⋮         ⋮      ⋱
 -------- ---------------- -------- ------- ------------- --------- ------------
                                                    1 column and 19 rows omitted
Latent Variables:    [:ind60, :dem60, :dem65]
Observed Variables:  [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]

```

load the example data

```jldoctest high_level; output = false
data = example_data("political_democracy")

# output

75×11 DataFrame
 Row │ x1       x2       x3       y1       y2        y3        y4        y5    ⋯
     │ Float64  Float64  Float64  Float64  Float64   Float64   Float64   Float ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │ 4.44265  3.63759  2.55762     2.5    0.0       3.33333   0.0       1.25 ⋯
   2 │ 5.3845   5.06259  3.56808     1.25   0.0       3.33333   0.0       6.25
   3 │ 5.96101  6.25575  5.22443     7.5    8.8      10.0       9.19999   8.75
   4 │ 6.286    7.56786  6.2675      8.9    8.8      10.0       9.19999   8.90
   5 │ 5.86363  6.81892  4.57368    10.0    3.33333  10.0       6.66667   7.5  ⋯
   6 │ 5.53339  5.1358   3.89227     7.5    3.33333   6.66667   6.66667   6.25
   7 │ 5.30827  5.07517  3.31621     7.5    3.33333   6.66667   6.66667   5.0
   8 │ 5.34711  4.85203  4.26318     7.5    2.23333  10.0       1.49633   6.25
  ⋮  │    ⋮        ⋮        ⋮        ⋮        ⋮         ⋮         ⋮         ⋮  ⋱
  69 │ 4.52179  4.12713  2.11331     5.0    0.0       8.2       0.0       5.0  ⋯
  70 │ 4.65396  3.55535  1.88192     2.9    3.33333   6.66667   3.33333   2.5
  71 │ 4.47734  3.09104  1.98791     5.4   10.0       6.66667   3.33333   3.75
  72 │ 5.33754  5.63121  3.491       7.5    8.8      10.0       6.06667   7.5
  73 │ 6.12905  6.40357  5.0018      7.5    7.0      10.0       6.853     7.5  ⋯
  74 │ 5.00395  4.96284  3.97699    10.0    6.66667  10.0      10.0      10.0
  75 │ 4.48864  4.89784  2.86757     3.75   3.33333   0.0       0.0       1.25
                                                   4 columns and 60 rows omitted
```

and specify our model as

```jldoctest high_level; output = false
model = Sem(
    specification = partable,
    data = data
)

# output

[ Info: Your model is acyclic, specifying the A Matrix as either Upper or Lower Triangular can have great performance benefits.
Structural Equation Model
- Loss Functions
   SemML
- Fields
   observed:  SemObsCommon
   imply:     RAM
   diff:      SemDiffOptim
```

We can now fit the model via

```jldoctest high_level; output = false
model_fit = sem_fit(model)

# output

Fitted Structural Equation Model
================================
------------- Model ------------
Structural Equation Model
- Loss Functions
   SemML
- Fields
   observed:  SemObsCommon
   imply:     RAM
   diff:      SemDiffOptim

----- Optimization result ------
 * Status: success

 * Candidate solution
    Final objective value:     2.120543e+01

 * Found with
    Algorithm:     L-BFGS

 * Convergence measures
    |x - x'|               = 3.81e-05 ≰ 1.5e-08
    |x - x'|/|x'|          = 5.10e-06 ≰ 0.0e+00
    |f(x) - f(x')|         = 1.05e-09 ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 4.97e-11 ≤ 1.0e-10
    |g(x)|                 = 7.31e-05 ≰ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    136
    f(x) calls:    413
    ∇f(x) calls:   413
```

and compute fit measures as

```jldoctest high_level
fit_measures(model_fit)

# output

Dict{Symbol, Union{Missing, Float64}} with 8 entries:
  :minus2ll => 3106.66
  :AIC      => 3168.66
  :BIC      => 3240.5
  :df       => 35.0
  :χ²       => 37.6169
  :p_value  => 0.350263
  :RMSEA    => 0.0315739
  :n_par    => 31.0
```

We can also get a bit more information about the fitted model via the `sem_summary()` function:

```jldoctest high_level
sem_summary(model_fit)

# output

Fitted Structural Equation Model

--------------------------------- Properties ---------------------------------

Optimization algorithm:      L-BFGS
Convergence:                 true
No. iterations/evaluations:  136

Number of parameters:        31
Number of observations:      75.0

----------------------------------- Model -----------------------------------

Structural Equation Model
- Loss Functions
   SemML
- Fields
   observed:  SemObsCommon
   imply:     RAM
   diff:      SemDiffOptim
```

To investigate the parameter estimates, we can update our `partable` object to contain the new estimates:

```jldoctest high_level; output = false
update_estimate!(partable, model_fit)

# output

 -------- ---------------- -------- ------- ------------- --------- ------------
    from   parameter_type       to    free   value_fixed     start    estimate ⋯
  Symbol           Symbol   Symbol    Bool       Float64   Float64     Float64 ⋯
 -------- ---------------- -------- ------- ------------- --------- ------------
   ind60                →       x1   false           1.0       0.0         0.0 ⋯
   ind60                →       x2    true           0.0       0.0     2.18037 ⋯
   ind60                →       x3    true           0.0       0.0      1.8185 ⋯
   dem60                →       y1   false           1.0       0.0         0.0 ⋯
   dem60                →       y2    true           0.0       0.0     1.25677 ⋯
   dem60                →       y3    true           0.0       0.0     1.05773 ⋯
   dem60                →       y4    true           0.0       0.0     1.26483 ⋯
   dem65                →       y5   false           1.0       0.0         0.0 ⋯
   dem65                →       y6    true           0.0       0.0     1.18571 ⋯
   dem65                →       y7    true           0.0       0.0     1.27951 ⋯
   dem65                →       y8    true           0.0       0.0     1.26598 ⋯
   ind60                →    dem60    true           0.0       0.0     1.48304 ⋯
   dem60                →    dem65    true           0.0       0.0    0.837369 ⋯
   ind60                →    dem65    true           0.0       0.0     0.57228 ⋯
      x1                ↔       x1    true           0.0       0.0   0.0826519 ⋯
    ⋮            ⋮            ⋮        ⋮          ⋮           ⋮          ⋮     ⋱
 -------- ---------------- -------- ------- ------------- --------- ------------
                                                    1 column and 19 rows omitted
Latent Variables:    [:ind60, :dem60, :dem65]
Observed Variables:  [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]

```

and investigate the solution with

```jldoctest high_level
sem_summary(partable)

# output

--------------------------------- Variables ---------------------------------

Latent variables:    ind60 dem60 dem65
Observed variables:  x1 x2 x3 y1 y2 y3 y4 y5 y6 y7 y8

---------------------------- Parameter Estimates -----------------------------

Loadings:

ind60

  to   estimate   identifier   value_fixed   start   parameter_type   from    free

  x1   0.0        const        1.0           0.0     →                ind60   0.0
  x2   2.18       θ_1          0.0           0.0     →                ind60   1.0
  x3   1.82       θ_2          0.0           0.0     →                ind60   1.0

dem60

  to   estimate   identifier   value_fixed   start   parameter_type   from    free

  y1   0.0        const        1.0           0.0     →                dem60   0.0
  y2   1.26       θ_3          0.0           0.0     →                dem60   1.0
  y3   1.06       θ_4          0.0           0.0     →                dem60   1.0
  y4   1.26       θ_5          0.0           0.0     →                dem60   1.0

dem65

  to   estimate   identifier   value_fixed   start   parameter_type   from    free

  y5   0.0        const        1.0           0.0     →                dem65   0.0
  y6   1.19       θ_6          0.0           0.0     →                dem65   1.0
  y7   1.28       θ_7          0.0           0.0     →                dem65   1.0
  y8   1.27       θ_8          0.0           0.0     →                dem65   1.0

Directed Effects:

  from        to      estimate   identifier   value_fixed   start   free

  ind60   →   dem60   1.48       θ_9          0.0           0.0     1.0
  dem60   →   dem65   0.84       θ_10         0.0           0.0     1.0
  ind60   →   dem65   0.57       θ_11         0.0           0.0     1.0

Variances:

  from        to      estimate   identifier   value_fixed   start   free

  x1      ↔   x1      0.08       θ_12         0.0           0.0     1.0
  x2      ↔   x2      0.12       θ_13         0.0           0.0     1.0
  x3      ↔   x3      0.47       θ_14         0.0           0.0     1.0
  y1      ↔   y1      1.92       θ_15         0.0           0.0     1.0
  y2      ↔   y2      7.47       θ_16         0.0           0.0     1.0
  y3      ↔   y3      5.14       θ_17         0.0           0.0     1.0
  y4      ↔   y4      3.19       θ_18         0.0           0.0     1.0
  y5      ↔   y5      2.38       θ_19         0.0           0.0     1.0
  y6      ↔   y6      5.02       θ_20         0.0           0.0     1.0
  y7      ↔   y7      3.48       θ_21         0.0           0.0     1.0
  y8      ↔   y8      3.3        θ_22         0.0           0.0     1.0
  ind60   ↔   ind60   0.45       θ_23         0.0           0.0     1.0
  dem60   ↔   dem60   4.01       θ_24         0.0           0.0     1.0
  dem65   ↔   dem65   0.17       θ_25         0.0           0.0     1.0

Covariances:

  from       to   estimate   identifier   value_fixed   start   free

  y1     ↔   y5   0.63       θ_26         0.0           0.0     1.0
  y2     ↔   y4   1.33       θ_27         0.0           0.0     1.0
  y2     ↔   y6   2.18       θ_28         0.0           0.0     1.0
  y3     ↔   y7   0.81       θ_29         0.0           0.0     1.0
  y8     ↔   y4   0.35       θ_30         0.0           0.0     1.0
  y8     ↔   y6   1.37       θ_31         0.0           0.0     1.0
```

Congratulations, you fitted and inspected your very first model! To learn more about the different parts, 
you may visit the sections on model specification (XXX), model construction (XXX), model fitting (XXX) and
model inspection (XXX).

If you want to learn how to extend the package (e.g., add a new loss function), you may visit the developer documentation (XXX).