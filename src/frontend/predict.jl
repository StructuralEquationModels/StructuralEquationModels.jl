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

# wrapper that materializes A and S matrices from the model params
function latent_scores_operator(
    ::Type{T},
    model::SemLoss,
    params::AbstractVector;
    kwargs...,
) where {T <: SemScoresPredictMethod}
    ram = SEM.implied(model).ram_matrices
    latent_scores_operator(
        T,
        SEM.implied(model),
        materialize(ram.A, params),
        materialize(ram.S, params);
        kwargs...,
    )
end

function latent_scores_operator(
    ::SemRegressionScores,
    implied::SemImplied,
    A::AbstractMatrix,
    S::AbstractMatrix;
    alpha::Number = 0,
)
    implied = SEM.implied(model)
    ram = implied.ram_matrices
    lv_inds = latent_var_indices(ram)

    lv_FA = ram.F * A[:, lv_inds]

    cov_lv = if alpha == 0
        lv_I_A⁻¹ = inv(I - A)[lv_inds, :]
        X_A_Xt(S, lv_I_A⁻¹)
    else
        inv(Xt_A_X(inv(S), I - A) + alpha * I)[lv_inds, lv_inds]
    end
    Σ = implied.Σ
    Σ⁻¹ = inv(Σ)
    return cov_lv * lv_FA' * Σ⁻¹
end

function latent_scores_operator(
    ::SemBartlettScores,
    implied::SemImplied,
    A::AbstractMatrix,
    S::AbstractMatrix;
    alpha::Number = 0,
)
    ram = implied.ram_matrices
    lv_inds = latent_var_indices(ram)
    lv_FA = ram.F * A[:, lv_inds]

    obs_inds = observed_var_indices(ram)
    ov_S⁻¹ = inv(S[obs_inds, obs_inds])
    cov_lv⁻¹ = Xt_A_X(ov_S⁻¹, lv_FA)
    (alpha != 0) && (cov_lv⁻¹ += alpha * I)
    cov_lv = inv(cov_lv⁻¹)

    return cov_lv * lv_FA' * ov_S⁻¹
end

function predict_latent_scores(
    method::SemScoresPredictMethod,
    model::SemLoss,
    params::AbstractVector,
    data::SemObserved = observed(model);
    alpha::Number = 0,
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
    (alpha < 0) &&
        throw(ArgumentError("The regularization parameter alpha must be non-negative"))

    implied = SEM.implied(model)
    ram = implied.ram_matrices
    update!(EvaluationTargets(0.0, nothing, nothing), implied, params)

    A = materialize(ram.A, params)
    S = materialize(ram.S, params)
    lv_inds = latent_var_indices(ram)
    lv_scores_op = latent_scores_operator(method, implied, A, S; alpha)

    data =
        data.data .- (isnothing(data.obs_mean) ? mean(data.data, dims = 1) : data.obs_mean')
    lv_scores = data * lv_scores_op'
    # adjust the scores w.r.t the variable means
    if MeanStruct(implied) === HasMeanStruct
        M = materialize(ram.M, params)
        lv_I_A⁻¹ = inv(I - A)[lv_inds, :]
        lv_scores .+= (lv_I_A⁻¹ * M)'
    end

    return lv_scores
end

predict_latent_scores(
    model::SemLoss,
    params::AbstractVector,
    data::SemObserved = observed(model);
    method::Symbol = :regression,
    alpha::Number = 0,
) = predict_latent_scores(SemScoresPredictMethod(method), model, params, data; alpha)
