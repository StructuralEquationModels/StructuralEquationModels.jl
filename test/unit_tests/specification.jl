@testset "ParameterTable - RAMMatrices conversion" begin
    partable = ParameterTable(ram_matrices)
    @test ram_matrices == RAMMatrices(partable)
end

@test params_to_indices([:x2, :x10, :x28], model_ml) == [2, 10, 28]

@testset "params_to_indices" begin
    pars = [:θ_1, :θ_7, :θ_21]
    @test params_to_indices(pars, model_ml) == params_to_indices(pars, partable)
    @test params_to_indices(pars, model_ml) ==
          params_to_indices(pars, RAMMatrices(partable))
end

# from docstrings:
param_indices = params_to_indices([:λ₁, λ₂], my_fitted_sem)
values = solution(my_fitted_sem)[param_indices]

graph = @StenoGraph begin
    # measurement model
    visual → fixed(1.0, 1.0) * x1 + fixed(0.5, 0.5) * x2 + fixed(0.6, 0.8) * x3
    textual → fixed(1.0, 1.0) * x4 + x5 + label(:a₁, :a₂) * x6
    speed → fixed(1.0, 1.0) * x7 + fixed(1.0, NaN) * x8 + label(:λ₉, :λ₉) * x9
    # variances and covariances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)
    visual ↔ textual + speed
    textual ↔ speed
end
