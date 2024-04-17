using Test, SafeTestsets, JuliaFormatter, StructuralEquationModels

@testset "JuliaFormatter.jl" begin
    @test format(StructuralEquationModels; verbose = false, overwrite = false)
end
@time @safetestset "Unit Tests" begin
    include("unit_tests/unit_tests.jl")
end
@time @safetestset "Example Models" begin
    include("examples/examples.jl")
end

if !haskey(ENV, "JULIA_EXTENDED_TESTS") || ENV["JULIA_EXTENDED_TESTS"] == "true"
    
end