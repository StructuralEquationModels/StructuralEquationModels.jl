using sem, Test, SafeTestsets

@time @safetestset "Examples" begin include("examples/example_tests.jl") end
