@testset "ParameterTable - RAMMatrices conversion" begin
    partable = ParameterTable(ram_matrices)
    @test ram_matrices == RAMMatrices(partable)
end

@testset "params()" begin
    @test params(model_ml)[2, 10, 28] == [:x2, :x10, :x28]
    @test params(model_ml) == params(partable)
    @test params(model_ml) == params(RAMMatrices(partable))
end

graph = @StenoGraph begin
    # measurement model
    visual  → fixed(1.0, 1.0)*x1 + fixed(0.5,     0.5)*x2 + fixed(0.6, 0.8)*x3
    textual → fixed(1.0, 1.0)*x4 +                     x5 + label(:a₁, :a₂)*x6
    speed   → fixed(1.0, 1.0)*x7 + fixed(1.0,     NaN)*x8 + label(:λ₉, :λ₉)*x9
    # variances and covariances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars)   ↔ _(latent_vars)
    visual ↔ textual + speed
    textual ↔ speed
end