using StructuralEquationModels
graph = @StenoGraph begin
    a → b
end
partable = ParameterTable(graph, observed_vars = [:a, :b], latent_vars = Symbol[])
update_partable!(partable, :estimate, param_labels(partable), [3.1415])
@testset "params" begin
    out = [NaN]
    StructuralEquationModels.params!(out, partable)
    @test params(partable) == out == [3.1415]
end
@testset "param_labels" begin
    @test param_labels(partable) == [:θ_1]
end

