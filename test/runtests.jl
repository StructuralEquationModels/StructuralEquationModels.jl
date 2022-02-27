using Test, SafeTestsets

@time @safetestset "Unit Tests" begin include("unit_tests/unit_tests.jl") end
@time @safetestset "Example Models" begin include("examples/example_tests.jl") end