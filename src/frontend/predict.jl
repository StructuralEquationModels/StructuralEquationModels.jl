function predict(
    model::SemLoss,
    params::AbstractVector,
    scores::AbstractMatrix;
    score_vars::Union{AbstractVector, Nothing} = latent_var_indices(model),
    predict_vars::Union{AbstractVector, Nothing} = observed_var_indices(model),
)
    ram = imply(model).ram_matrices
    score_var_inds =
        !isnothing(score_vars) ? check_var_indices(ram, score_vars, normalize = false) :
        nothing
    A = materialize(ram.A, params)
    I_A⁻¹ = inv(I - A)
    sv_I_A⁻¹ = !isnothing(score_var_inds) ? I_A⁻¹[:, score_var_inds] : I_A⁻¹
    res = scores * sv_I_A⁻¹'
    if MeanStructure(imply(model)) === HasMeanStructure
        # score_vars intercepts already included in the scores
        M = materialize(ram.M, params)
        if isnothing(score_var_inds)
            fill!(M, 0)
        else
            M[score_var_inds, :] .= 0
        end
        res .+= (I_A⁻¹ * M)'
    end
    if !isnothing(predict_vars)
        predict_var_inds = check_var_indices(ram, predict_vars, normalize = false)
        res = res[:, predict_var_inds]
    end
    return res
end

"""
    SemScoresPredictMethod

Abstract supertype for latent-score prediction methods used by [`predict_latent_scores`](@ref).

The subtypes should implement the [`latent_scores_solver()`](@ref) method that creates
an instance of [`ScoresSolver`](@ref).

*SEM.jl* implements the following methods:

- [`SemRegressionScores`](@ref) for *Regression/Thomson* scores.
- [`SemBartlettScores`](@ref) for *Bartlett* scores.
- [`SemAndersonRubinScores`](@ref) for *Anderson-Rubin* scores.

See the concrete type docstrings for the mathematical definitions and interpretation of
each method.
"""
abstract type SemScoresPredictMethod end

raw"""
    SemRegressionScores

*Regression/Thomson* latent scores.

For centered observed data vector `y`, observed loading matrix `Λ`, observed residual
covariance `Ψ`, and marginal covariance `Φ` of the selected latent variables, the
regression scores are

```math
\hat z_{\mathrm{reg}}
= \Phi \Lambda^\top \Sigma^{-1} y,
\qquad
\Sigma = \Lambda \Phi \Lambda^\top + \Psi,
```

which is equivalent to the penalized least-squares problem

```math
\hat z_{\mathrm{reg}}
= \arg\min_z \left(\lVert L_\Psi (y - \Lambda z) \rVert_2^2
+ \lVert L_\Phi z \rVert_2^2
+ \alpha \lVert z \rVert_2^2 \right),
```

where `L_Ψ' L_Ψ = Ψ^{-1}` and `L_Φ' L_Φ = Φ^{-1}`. The prior term `\lVert L_\Phi z \rVert_2^2`
shrinks `z` toward the latent mean, penalizing directions with small prior variance more
strongly than directions with large prior variance. Internally the scores are computed via
an augmented QR solve instead of explicitly forming the normal equations.

# References

1. G. H. Thomson, *The Factorial Analysis of Human Ability*, University of London Press,
    1939.
2. Penn State STAT 505, *12.12 - Estimation of Factor Scores*:
    https://online.stat.psu.edu/stat505/lesson/12/12.12
3. UCLA Statistical Methods and Data Analytics, *A Practical Introduction to Factor Analysis*:
    https://stats.oarc.ucla.edu/spss/seminars/introduction-to-factor-analysis/
"""
struct SemRegressionScores <: SemScoresPredictMethod end

raw"""
    SemBartlettScores

*Bartlett* latent scores.

For centered observed data vector `y`, observed loading matrix `Λ`, and observed
residual covariance `Ψ`, the Bartlett scores are

```math
\hat z_{\mathrm{Bartlett}}
= \left(\Lambda^\top \Psi^{-1} \Lambda + \alpha I\right)^{-1}
  \Lambda^\top \Psi^{-1} y,
```

which is equivalent to the weighted ridge least-squares problem

```math
\hat z_{\mathrm{Bartlett}}
= \arg\min_z \left(\lVert L_\Psi (y - \Lambda z) \rVert_2^2
+ \alpha \lVert z \rVert_2^2 \right),
```

where `L_Ψ' L_Ψ = Ψ^{-1}`. Internally the score operator is computed from the
augmented QR solve instead of explicitly forming the normal equations.

# References

1. M. S. Bartlett, *The Statistical Conception of Mental Factors*, British Journal of
    Psychology, 28(1), 97-104, 1937.
2. Penn State STAT 505, *12.12 - Estimation of Factor Scores*:
    https://online.stat.psu.edu/stat505/lesson/12/12.12
3. UCLA Statistical Methods and Data Analytics, *A Practical Introduction to Factor Analysis*:
    https://stats.oarc.ucla.edu/spss/seminars/introduction-to-factor-analysis/
"""
struct SemBartlettScores <: SemScoresPredictMethod end

raw"""
        SemAndersonRubinScores

*Anderson-Rubin* latent scores.

Let

```math
B_{\mathrm{Bartlett}}
= \left(\Lambda^\top \Psi^{-1} \Lambda + \alpha I\right)^{-1}
    \Lambda^\top \Psi^{-1}
```

be the Bartlett score operator and let `Σ` be the model-implied observed covariance.
Define the model-implied covariance of the Bartlett scores by

```math
C_{\mathrm{Bartlett}}
= B_{\mathrm{Bartlett}} \Sigma B_{\mathrm{Bartlett}}^\top.
```

The Anderson-Rubin operator is the standardized Bartlett operator

```math
B_{\mathrm{AR}} = C_{\mathrm{Bartlett}}^{-1/2} B_{\mathrm{Bartlett}},
```

so the resulting scores satisfy

```math
\operatorname{Cov}_{\Sigma}(\hat z_{\mathrm{AR}}) = I.
```

Equivalently, Anderson-Rubin scores are obtained in two steps:

```math
\widetilde{z}
= \arg\min_z \left(\lVert L_\Psi (y - \Lambda z) \rVert_2^2
+ \alpha \lVert z \rVert_2^2 \right),
```

followed by the whitening transform

```math
\hat z_{\mathrm{AR}} = C_{\mathrm{Bartlett}}^{-1/2} \widetilde{z}.
```

!!! warning "Different latent basis"
    *Anderson-Rubin* scores live in a whitened basis of the latent subspace, not
    generally in the original latent-variable basis parameterized by ``Φ``. Unless the
    whitening transform is diagonal or identity, each reported Anderson-Rubin score is
    a linear combination of the model's latent variables. If scores on the original
    latent-variable scale are needed, prefer [`SemRegressionScores`](@ref) or
    [`SemBartlettScores`](@ref).

For correlated latent variables, these scores are standardized and orthogonalized, so they
do not preserve the original latent covariance ``Φ``; instead they return whitened Bartlett
coordinates with identity model-implied covariance.

# References

1. T. W. Anderson and H. Rubin, *Statistical Inference in Factor Analysis*, in
    *Proceedings of the Third Berkeley Symposium on Mathematical Statistics and
    Probability*, volume 5, 1956, pp. 111-150.
2. C. DiStefano, M. Zhu, and D. Mindrila, *Understanding and Using Factor Scores:
    Considerations for the Applied Researcher*, Practical Assessment, Research and
    Evaluation, 14(20), 2009.
3. UCLA Statistical Methods and Data Analytics, *A Practical Introduction to Factor Analysis*:
    https://stats.oarc.ucla.edu/spss/seminars/introduction-to-factor-analysis/
"""
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
    latent_vars::AbstractVector,
    params::AbstractVector;
    kwargs...,
) where {T <: SemScoresPredictMethod}
    ram = SEM.implied(model).ram_matrices
    latent_scores_operator(
        T,
        SEM.implied(model),
        latent_vars,
        materialize(ram.A, params),
        materialize(ram.S, params);
        kwargs...,
    )
end

function latent_scores_operator(
    ::SemRegressionScores,
    implied::SemImplied,
    latent_vars::AbstractVector,
    A::AbstractMatrix,
    S::AbstractMatrix;
    alpha::Number = 0,
)
    implied = SEM.implied(model)
    ram = implied.ram_matrices

    lv_FA = ram.F * A[:, latent_vars]

    cov_lv = if alpha == 0
        lv_I_A⁻¹ = inv(I - A)[latent_vars, :]
        X_A_Xt(S, lv_I_A⁻¹)
    else
        inv(Xt_A_X(inv(S), I - A) + alpha * I)[latent_vars, latent_vars]
    end
    Σ = implied.Σ
    Σ⁻¹ = inv(Σ)
    return cov_lv * lv_FA' * Σ⁻¹
end

function latent_scores_operator(
    ::SemBartlettScores,
    implied::SemImplied,
    latent_vars::AbstractVector,
    A::AbstractMatrix,
    S::AbstractMatrix;
    alpha::Number = 0,
)
    ram = implied.ram_matrices
    lv_FA = ram.F * A[:, latent_vars]

    obs_inds = observed_var_indices(ram)
    ov_S⁻¹ = inv(S[obs_inds, obs_inds])
    cov_lv⁻¹ = Xt_A_X(ov_S⁻¹, lv_FA)
    (alpha != 0) && (cov_lv⁻¹ += alpha * I)
    cov_lv = inv(cov_lv⁻¹)

    return cov_lv * lv_FA' * ov_S⁻¹
end

"""
    predict_latent_scores(
        model::SemLoss, params, data = observed(model);
        method = :regression, latent_vars = nothing, alpha = 0
    )

Predict latent scores for the selected latent variables from observed data.

`method` selects the latent-score definition. See [`SemScoresPredictMethod`](@ref) and
its concrete implementations [`SemRegressionScores`](@ref),
[`SemBartlettScores`](@ref), and [`SemAndersonRubinScores`](@ref) for the mathematical
definitions and interpretation of each method.

`latent_vars` selects which latent variables are scored. If `latent_vars = nothing`, all
latent variables in the model are scored.

`alpha` is a non-negative ridge regularization parameter passed to the selected scoring method.
"""
function predict_latent_scores(
    method::SemScoresPredictMethod,
    model::SemLoss,
    params::AbstractVector,
    data::SemObserved = observed(model);
    latent_vars::Union{AbstractVector, Nothing} = nothing,
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
    lv_inds = check_var_indices(ram, latent_vars, allow_observed = false, normalize = true)

    update!(EvaluationTargets(0.0, nothing, nothing), implied, params)

    A = materialize(ram.A, params)
    S = materialize(ram.S, params)
    lv_scores_op = latent_scores_operator(method, implied, lv_inds, A, S; alpha)

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
    latent_vars::Union{AbstractVector, Nothing} = nothing,
    alpha::Number = 0,
) = predict_latent_scores(
    SemScoresPredictMethod(method),
    model,
    params,
    data;
    latent_vars,
    alpha,
)
