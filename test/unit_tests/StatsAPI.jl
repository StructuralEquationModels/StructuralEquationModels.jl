using StructuralEquationModels
graph = @StenoGraph begin
    a → b
end
partable = ParameterTable(graph, observed_vars = [:a, :b], latent_vars = Symbol[])
update_partable!(partable, :estimate, param_labels(partable), [3.1415])
data = randn(100, 2)
model = Sem(
    specification = partable,
    data = data
)
model_fit = fit(model)

@testset "params" begin
    out = [NaN]
    StructuralEquationModels.params!(out, partable)
    @test params(partable) == out == [3.1415] == coef(partable)
end
@testset "param_labels" begin
    @test param_labels(partable) == [:θ_1] == coefnames(partable)
end

@testset "nobs" begin
    @test nobs(model) == nsample(model)
end

@testset "coeftable" begin
    @test_throws coeftable(model) MethodError "StructuralEquationModels does not support"
end