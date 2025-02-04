abstract type SemScoresPredictMethod end

struct SemRegressionScores <: SemScoresPredictMethod end
struct SemBartlettScores <: SemScoresPredictMethod end
struct SemAndersonRubinScores <: SemScoresPredictMethod end

function SemScoresPredictMethod(method::Symbol)
    if method == :regression
        return SemRegressionScores()
    elseif method == :Bartlett
        return SemBartlettScores()
    elseif method == :AndersonRubin
        return SemAndersonRubinScores()
    else
        throw(ArgumentError("Unsupported prediction method: $method"))
    end
end

predict_latent_scores(
    fit::SemFit,
    data::SemObserved = observed(sem_term(fit.model));
    method::Symbol = :regression,
) = predict_latent_scores(SemScoresPredictMethod(method), fit, data)

predict_latent_scores(
    method::SemScoresPredictMethod,
    fit::SemFit,
    data::SemObserved = observed(sem_term(fit.model)),
) = predict_latent_scores(method, loss(sem_term(fit.model)), fit.solution, data)

function inv_cov!(A::AbstractMatrix)
    if istril(A)
        A = LowerTriangular(A)
    elseif istriu(A)
        A = UpperTriangular(A)
    else
    end
    A_chol = Cholesky(A)
    return inv!(A_chol)
end

function latent_scores_operator(
    ::SemRegressionScores,
    model::SemLoss,
    params::AbstractVector,
)
    implied = SEM.implied(model)
    ram = implied.ram_matrices
    lv_inds = latent_var_indices(ram)

    A = materialize(ram.A, params)
    lv_FA = ram.F * A[:, lv_inds]
    lv_I_A⁻¹ = inv(I - A)[lv_inds, :]

    S = materialize(ram.S, params)

    cov_lv = lv_I_A⁻¹ * S * lv_I_A⁻¹'
    Σ = implied.Σ
    Σ⁻¹ = inv(Σ)
    return cov_lv * lv_FA' * Σ⁻¹
end

function latent_scores_operator(::SemBartlettScores, model::SemLoss, params::AbstractVector)
    implied = SEM.implied(model)
    ram = implied.ram_matrices
    lv_inds = latent_var_indices(ram)
    A = materialize(ram.A, params)
    lv_FA = ram.F * A[:, lv_inds]

    S = materialize(ram.S, params)
    obs_inds = observed_var_indices(ram)
    ov_S⁻¹ = inv(S[obs_inds, obs_inds])

    return inv(lv_FA' * ov_S⁻¹ * lv_FA) * lv_FA' * ov_S⁻¹
end

function predict_latent_scores(
    method::SemScoresPredictMethod,
    model::SemLoss,
    params::AbstractVector,
    data::SemObserved = observed(model),
)
    nobserved_vars(data) == nobserved_vars(model) || throw(
        DimensionMismatch(
            "Number of variables in data ($(nsamples(data))) does not match the number of observed variables in the model ($(nobserved_vars(model)))",
        ),
    )
    length(params) == nparams(model) || throw(
        DimensionMismatch(
            "The length of parameters vector ($(length(params))) does not match the number of parameters in the model ($(nparams(model)))",
        ),
    )

    implied = SEM.implied(model)
    update!(EvaluationTargets(0.0, nothing, nothing), implied, params)
    ram = implied.ram_matrices
    lv_inds = latent_var_indices(ram)
    A = materialize(ram.A, params)
    lv_I_A⁻¹ = inv(I - A)[lv_inds, :]

    lv_scores_op = latent_scores_operator(method, model, params)

    data =
        data.data .- (isnothing(data.obs_mean) ? mean(data.data, dims = 1) : data.obs_mean')
    lv_scores = data * lv_scores_op'
    if MeanStructure(implied) === HasMeanStructure
        M = materialize(ram.M, params)
        lv_scores .+= (lv_I_A⁻¹ * M)'
    end

    return lv_scores
end

predict_latent_scores(
    model::SemLoss,
    params::AbstractVector,
    data::SemObserved = observed(model);
    method::Symbol = :regression,
) = predict_latent_scores(SemScoresPredictMethod(method), model, params, data)
