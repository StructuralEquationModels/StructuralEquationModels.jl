using Test, SafeTestsets, JuliaFormatter, StructuralEquationModels

@testset "JuliaFormatter.jl" begin
    if !format(StructuralEquationModels; verbose = false, overwrite = false)
        @warn "Julia code formatting style inconsistencies detected."
    end
end
@time @safetestset "Unit Tests" begin
    include("unit_tests/unit_tests.jl")
end
@time @safetestset "Example Models" begin
    include("examples/examples.jl")
end
