# Parametric types

Recall that a new composite type in julia can be declared as

```julia
struct MyNewType
    field1
    field2
    ...
end
```

Often we can speedup computations by declaring our type as a **Parametric Type**:

```julia
struct MyNewType{A, B}
    field1::A
    field2::B
    ...
end
```

giving each field a type and adding them as parameters to our type declaration.

Recall our example from [Custom loss functions](@ref):

```julia
struct Ridge <: SemLossFunction
    α
    I
end
```

We could also declare it as a parametric type:

```julia
struct ParametricRidge{X, Y} <: SemLossFunction
    α::X
    I::Y
end
```

Let's see how this might affect performance:

```julia
function add_α(ridge1, ridge2)
    return ridge1.α + ridge2.α 
end

my_ridge_1 = Ridge(2.5, [2,3])
my_ridge_2 = Ridge(25.38, [2,3])

my_parametric_ridge_1 = ParametricRidge(2.1, [2,3])
my_parametric_ridge_2 = ParametricRidge(8.34, [2,3])

using BenchmarkTools

@benchmark add_α($my_ridge_1, $my_ridge_2)

# output

BenchmarkTools.Trial: 10000 samples with 994 evaluations.
 Range (min … max):  16.073 ns …  1.508 μs  ┊ GC (min … max): 0.00% … 98.35%
 Time  (median):     17.839 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   22.846 ns ± 23.564 ns  ┊ GC (mean ± σ):  1.64% ±  1.70%

@benchmark add_α($my_parametric_ridge_1, $my_parametric_ridge_2)

# output

BenchmarkTools.Trial: 10000 samples with 1000 evaluations.
 Range (min … max):  1.371 ns … 20.250 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     2.097 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.169 ns ±  0.829 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

```

which is quite a difference. To learn more about parametric types, see the [this section](https://docs.julialang.org/en/v1/manual/types/#Parametric-Types) in the julia documentation.