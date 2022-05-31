# Build by parts

You can always build a model by parts - that is, you construct the observed, imply, loss and diff part seperately first.

As an example on how this works, we will build [A first model](@ref) in parts.

First, we define the model just as usual:

```@example build
using StructuralEquationModels

data = example_data("political_democracy")

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
```

Now, we construct the different parts:

```@example build
# observed ---------------------------------------------------------------------------------
observed = SemObsData(specification = partable, data = data)

# imply ------------------------------------------------------------------------------------
imply_ram = RAM(specification = partable)

# loss -------------------------------------------------------------------------------------
ml = SemML(observed = observed)

loss_ml = SemLoss(ml)

# diff -------------------------------------------------------------------------------------
diff_obj = SemDiffOptim()

# model ------------------------------------------------------------------------------------

model_ml = Sem(observed, imply_ram, loss_ml, diff_obj)
```

Different models may need additional arguments:

```@example build
model_ml_fd = SemFiniteDiff(observed, imply_ram, loss_ml, diff_obj, Val(false))
```