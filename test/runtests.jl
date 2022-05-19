using Test, SafeTestsets

@test ENV["JULIA_EXTENDED_TESTS"] == "true"
@time @safetestset "Unit Tests" begin include("unit_tests/unit_tests.jl") end
@time @safetestset "Example Models" begin include("examples/examples.jl") end