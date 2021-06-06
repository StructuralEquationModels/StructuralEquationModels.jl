using sem, Test, SafeTestsets

@safetestset "Koch" begin include("example_koch.jl") end
@safetestset "DefVars" begin include("example_defvars.jl") end
