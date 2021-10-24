using Test, SafeTestsets

@safetestset "Political Democracy" begin include("political_democracy.jl") end
#@safetestset "DefVars" begin include("example_defvars.jl") end