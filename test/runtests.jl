using Test, SafeTestsets

@time @safetestset "Example Models" begin include("examples/example_tests.jl") end