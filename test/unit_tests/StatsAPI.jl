@testset "StatsAPI" begin
    using StructuralEquationModels
    graph = @StenoGraph begin
        a →  b
    end
    partable = ParameterTable(graph, observed_vars = [:a, :b], latent_vars = Symbol[])
    @testset "params" begin
        out = [NaN]
        StructuralEquationModels.params!(out, partable)
        @test params(partable) == out == [NaN]
    end
    @testset "param_labels" begin
        @test param_labels(partable) == [:θ_1]
    end
end
