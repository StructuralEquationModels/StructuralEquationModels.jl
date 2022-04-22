@testset "ParameterTable - RAMMatrices conversion" begin
    partable = ParameterTable(ram_matrices)
    @test ram_matrices == RAMMatrices(partable)
end

@test get_identifier_indices([:x2, :x10, :x28], model_ml) == [2, 10, 28]
