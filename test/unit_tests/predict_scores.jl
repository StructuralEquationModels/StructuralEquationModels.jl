using StructuralEquationModels, Test, LinearAlgebra, Statistics, Logging

SEM = StructuralEquationModels

function inverse_rows(A::AbstractMatrix, row_inds::AbstractVector{<:Integer})
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square."))
    rhs = zeros(eltype(A), n, length(row_inds))
    for (j, row_ix) in pairs(row_inds)
        rhs[row_ix, j] = one(eltype(A))
    end
    return copy((A' \ rhs)')
end

function bartlett_operator(Λ::AbstractMatrix, Ψ::AbstractMatrix; alpha::Real = 0.0)
    nobs, nlat = size(Λ)
    T = float(promote_type(eltype(Λ), eltype(Ψ), typeof(alpha)))
    Ψ_chol = cholesky(Symmetric(Matrix{T}(Ψ)))
    I_obs = Matrix{T}(I, nobs, nobs)
    I_lat = Matrix{T}(I, nlat, nlat)
    Ψ⁻¹Λ = Ψ_chol \ Matrix{T}(Λ)
    Ψ⁻¹ = Ψ_chol \ I_obs
    return (Λ' * Ψ⁻¹Λ + T(alpha) * I_lat) \ (Λ' * Ψ⁻¹)
end

function regression_operator(
    Λ::AbstractMatrix,
    Ψ::AbstractMatrix,
    Φ::AbstractMatrix;
    alpha::Real = 0.0,
)
    nobs, nlat = size(Λ)
    T = float(promote_type(eltype(Λ), eltype(Ψ), eltype(Φ), typeof(alpha)))
    Ψ_chol = cholesky(Symmetric(Matrix{T}(Ψ)))
    Φ_chol = cholesky(Symmetric(Matrix{T}(Φ)))
    I_obs = Matrix{T}(I, nobs, nobs)
    I_lat = Matrix{T}(I, nlat, nlat)
    Ψ⁻¹Λ = Ψ_chol \ Matrix{T}(Λ)
    Ψ⁻¹ = Ψ_chol \ I_obs
    Φ⁻¹ = Φ_chol \ I_lat
    return (Λ' * Ψ⁻¹Λ + Φ⁻¹ + T(alpha) * I_lat) \ (Λ' * Ψ⁻¹)
end

function anderson_rubin_operator(
    Λ::AbstractMatrix,
    Ψ::AbstractMatrix,
    Σ::AbstractMatrix;
    alpha::Real = 0.0,
)
    B_bartlett = bartlett_operator(Λ, Ψ; alpha)
    C_bartlett = Symmetric(B_bartlett * Σ * B_bartlett')
    C_chol = cholesky(C_bartlett)
    return C_chol.U' \ B_bartlett
end

@testset "predict_latent_scores formulas" begin
    A = [
        0 0 0 0 1.0 0;
        0 0 0 0 :lambda21 0;
        0 0 0 0 0 1.0;
        0 0 0 0 0 :lambda42;
        0 0 0 0 0 0;
        0 0 0 0 0 0
    ]
    S = [
        :psi1 0 0 0 0 0;
        0 :psi2 0 0 0 0;
        0 0 :psi3 0 0 0;
        0 0 0 :psi4 0 0;
        0 0 0 0 :phi11 :phi12;
        0 0 0 0 :phi12 :phi22
    ]
    F = [
        1.0 0 0 0 0 0;
        0 1 0 0 0 0;
        0 0 1 0 0 0;
        0 0 0 1 0 0
    ]
    params_syms = [:lambda21, :lambda42, :psi1, :psi2, :psi3, :psi4, :phi11, :phi12, :phi22]
    spec = RAMMatrices(;
        A,
        S,
        F,
        param_labels = params_syms,
        vars = [:y1, :y2, :y3, :y4, :eta1, :eta2],
    )

    obs_colnames = [:y1, :y2, :y3, :y4]
    model = SemML(SemObservedData(data = randn(24, 4), observed_vars = obs_colnames), RAM(spec))
    data_values = [
        1.1 0.3 -0.4 0.8;
        0.7 -0.2 0.2 1.1;
        -0.3 -0.7 0.5 0.1;
        0.5 0.9 -0.1 -0.4
    ]
    data = SemObservedData(data = data_values, observed_vars = obs_colnames)
    params = [0.8, 0.9, 0.3, 0.4, 0.35, 0.45, 1.0, 0.2, 1.3]

    implied = SEM.implied(model)
    SEM.update!(SEM.EvaluationTargets(0.0, nothing, nothing), implied, params)

    ram = implied.ram_matrices
    lv_vars = [:eta1, :eta2]
    lv_inds = SEM.latent_var_indices(ram)
    obs_inds = SEM.observed_var_indices(ram)

    A_mtx = SEM.materialize(ram.A, params)
    S_mtx = SEM.materialize(ram.S, params)
    centered_data = data.data .- mean(data.data, dims = 1)
    Λ = Matrix(ram.F * A_mtx[:, lv_inds])
    Ψ = Matrix(S_mtx[obs_inds, obs_inds])
    I_A = Matrix{Float64}(I, size(A_mtx, 1), size(A_mtx, 2)) - A_mtx
    lv_I_A⁻¹ = inverse_rows(I_A, lv_inds)
    Φ = SEM.X_A_Xt(S_mtx, lv_I_A⁻¹)
    Σ = Matrix(implied.Σ)

    bartlett_alpha = 0.15
    bartlett_scores = SEM.predict_latent_scores(
        model,
        params,
        data;
        method = :Bartlett,
        latent_vars = lv_vars,
        alpha = bartlett_alpha,
    )
    bartlett_op = bartlett_operator(Λ, Ψ; alpha = bartlett_alpha)
    @test bartlett_scores ≈ centered_data * bartlett_op' rtol = 1e-10 atol = 1e-10

    regression_alpha = 0.2
    regression_scores = SEM.predict_latent_scores(
        model,
        params,
        data;
        method = :regression,
        latent_vars = lv_vars,
        alpha = regression_alpha,
    )
    regression_op = regression_operator(Λ, Ψ, Φ; alpha = regression_alpha)
    @test regression_scores ≈ centered_data * regression_op' rtol = 1e-10 atol = 1e-10

    regression_scores_0 = SEM.predict_latent_scores(
        model,
        params,
        data;
        method = :regression,
        latent_vars = lv_vars,
        alpha = 0.0,
    )
    Σ_chol = cholesky(Symmetric(Σ))
    regression_op_0 = Φ * Λ' * (Σ_chol \ Matrix{Float64}(I, size(Σ, 1), size(Σ, 2)))
    @test regression_scores_0 ≈ centered_data * regression_op_0' rtol = 1e-10 atol = 1e-10

    bartlett_scores_0 = SEM.predict_latent_scores(
        model,
        params,
        data;
        method = :Bartlett,
        latent_vars = lv_vars,
        alpha = 0.0,
    )
    @test !isapprox(regression_scores_0, bartlett_scores_0; rtol = 1e-6, atol = 1e-6)

    ar_alpha = 0.15
    ar_scores = SEM.predict_latent_scores(
        model,
        params,
        data;
        method = :AndersonRubin,
        latent_vars = lv_vars,
        alpha = ar_alpha,
    )
    ar_op = anderson_rubin_operator(Λ, Ψ, Σ; alpha = ar_alpha)
    @test ar_scores ≈ centered_data * ar_op' rtol = 1e-10 atol = 1e-10
    @test ar_op * Σ * ar_op' ≈ Matrix{Float64}(I, size(ar_op, 1), size(ar_op, 1)) rtol =
        1e-10 atol = 1e-10

    ar_scores_0 = SEM.predict_latent_scores(
        model,
        params,
        data;
        method = :AndersonRubin,
        latent_vars = lv_vars,
        alpha = 0.0,
    )
    @test ar_scores_warn ≈ ar_scores_0 rtol = 1e-10 atol = 1e-10
    @test !isapprox(ar_scores_0, bartlett_scores_0; rtol = 1e-6, atol = 1e-6)
end
