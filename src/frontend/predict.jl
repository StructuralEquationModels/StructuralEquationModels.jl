function predict(
    model::SemLoss,
    params::AbstractVector,
    scores::AbstractMatrix;
    score_vars::Union{AbstractVector, Nothing} = latent_var_indices(model),
    predict_vars::Union{AbstractVector, Nothing} = observed_var_indices(model),
)
    ram = implied(model).ram_matrices
    score_var_inds =
        !isnothing(score_vars) ? check_var_indices(ram, score_vars, normalize = false) :
        nothing
    A = materialize(ram.A, params)
    I_A⁻¹ = inv(I - A)
    sv_I_A⁻¹ = !isnothing(score_var_inds) ? I_A⁻¹[:, score_var_inds] : I_A⁻¹
    res = scores * sv_I_A⁻¹'
    if MeanStruct(implied(model)) === HasMeanStruct
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

# internal helper for calculating latent scores
# a callable object that recieves centered data matrix
# and returns the corresponding latent scores
abstract type ScoresSolver end

# solver that uses QR decomposition for numerical stability
struct QRScoresSolver{F, C} <: ScoresSolver
    factorization::F
    obs_chol::C
    extra_rhs_rows::Int
end

function QRScoresSolver(
    loadings::AbstractMatrix,
    obs_cov::AbstractMatrix;
    prior_cov::Union{AbstractMatrix, Nothing} = nothing,
    prior_cov_alpha::Number = 1,
    alpha::Number = 0,
)
    alpha >= 0 ||
        throw(ArgumentError("The regularization parameter alpha must be non-negative"))
    prior_cov_alpha >= 0 || throw(
        ArgumentError("The regularization parameter prior_cov_alpha must be non-negative"),
    )

    _, nlat = size(loadings)
    T = float(
        promote_type(
            eltype(loadings),
            eltype(obs_cov),
            isnothing(prior_cov) ? Float64 : eltype(prior_cov),
            typeof(prior_cov_alpha),
            typeof(alpha),
        ),
    )

    Λ = Matrix{T}(loadings)
    Ψ_chol = cholesky(Symmetric(Matrix{T}(obs_cov)))
    I_lat = Matrix{T}(I, nlat, nlat)

    aug_lhs = Matrix{T}[Matrix(Ψ_chol.U' \ Λ)]
    extra_rhs_rows = 0

    if !isnothing(prior_cov) && prior_cov_alpha != 0
        Φ_chol = cholesky(Symmetric(Matrix{T}(prior_cov)))
        push!(aug_lhs, sqrt(T(prior_cov_alpha)) * Matrix(Φ_chol.U' \ I_lat))
        extra_rhs_rows += nlat
    end

    if alpha != 0
        push!(aug_lhs, sqrt(T(alpha)) * I_lat)
        extra_rhs_rows += nlat
    end

    lhs_qr = qr(reduce(vcat, aug_lhs), ColumnNorm())
    return QRScoresSolver(lhs_qr, Ψ_chol, extra_rhs_rows)
end

function (solver::QRScoresSolver)(data::AbstractMatrix)
    T = float(promote_type(eltype(data), eltype(solver.obs_chol)))
    whitened_t = Matrix{T}((Matrix{T}(data) / solver.obs_chol.U)')
    rhs = if solver.extra_rhs_rows == 0
        whitened_t
    else
        vcat(whitened_t, zeros(T, solver.extra_rhs_rows, size(data, 1)))
    end
    return permutedims(solver.factorization \ rhs)
end

struct WhitenedScoresSolver{S <: ScoresSolver, C} <: ScoresSolver
    base_solver::S
    score_cov_chol::C
end

function (solver::WhitenedScoresSolver)(data::AbstractMatrix)
    base_scores = solver.base_solver(data)
    return base_scores / solver.score_cov_chol.U
end

const BasisTransformOperator = Union{AbstractMatrix, UniformScaling}

"""
    SemVariablesTransform

Linear change-of-basis metadata for centered latent scores.

`old_to_new` maps centered variables from the original basis to the transformed basis.
`new_to_old` applies the inverse map.

For [`SemAndersonRubinScores`](@ref), the method basis is the whitened Anderson-Rubin
coordinate system. For [`SemRegressionScores`](@ref) and [`SemBartlettScores`](@ref), the
transform is the identity.
"""
struct SemVariablesTransform{TN <: BasisTransformOperator, TO <: BasisTransformOperator}
    vars::Vector{Symbol}
    old_to_new::TN
    new_to_old::TO
end

"""
    (transform::SemVariablesTransform)(scores; inverse = false)

Apply a centered latent-score basis transform.

When `inverse = false` (default), the rows of `scores` are interpreted in the model's
original latent-variable basis and mapped to the method basis stored in `transform`.
When `inverse = true`, the rows are interpreted in the method basis and mapped back to
the original latent-variable basis.

The transform acts on centered scores. For uncentered scores, subtract the corresponding
latent means before applying the transform and add the target-basis means after.
"""
function (transform::SemVariablesTransform)(scores::AbstractMatrix; inverse::Bool = false)
    size(scores, 2) == length(transform.vars) || throw(
        DimensionMismatch(
            "Number of score columns ($(size(scores, 2))) does not match transform size ($(length(transform.vars))).",
        ),
    )
    basis_mtx = inverse ? transform.new_to_old : transform.old_to_new
    return scores * basis_mtx
end

function (transform::SemVariablesTransform)(scores::AbstractVector; inverse::Bool = false)
    length(scores) == length(transform.vars) || throw(
        DimensionMismatch(
            "Score vector length ($(length(scores))) does not match transform size ($(length(transform.vars))).",
        ),
    )
    return vec(transform(reshape(collect(scores), 1, :); inverse))
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

"""
    latent_scores_solver(
        method::SemScoresPredictMethod,
        implied::SemImplied,
        latent_vars::AbstractVector,
        A::AbstractMatrix, S::AbstractMatrix,
        lv_I_A⁻¹::Union{AbstractMatrix, Nothing} = nothing;
        alpha::Number = 0,
        prior_cov_alpha::Number = 1,
    ) -> ScoresSolver

Create a solver for the latent scores based on the specified prediction method.
"""
latent_scores_solver

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
+ \lambda_\Phi \lVert L_\Phi z \rVert_2^2
+ \alpha \lVert z \rVert_2^2 \right),
```

where `L_Ψ' L_Ψ = Ψ^{-1}` and `L_Φ' L_Φ = Φ^{-1}`. The classical regression/Thomson
scores correspond to `prior_cov_alpha = λ_Φ = 1`. Setting `prior_cov_alpha = 0`
switches off the latent-covariance prior and recovers the Bartlett objective with the
same `alpha`. Internally the scores are computed via an augmented QR solve instead of
explicitly forming the normal equations.

# References

1. G. H. Thomson, *The Factorial Analysis of Human Ability*, University of London Press,
    1939.
2. Penn State STAT 505, *12.12 - Estimation of Factor Scores*:
    https://online.stat.psu.edu/stat505/lesson/12/12.12
3. UCLA Statistical Methods and Data Analytics, *A Practical Introduction to Factor Analysis*:
    https://stats.oarc.ucla.edu/spss/seminars/introduction-to-factor-analysis/
"""
struct SemRegressionScores <: SemScoresPredictMethod end

function QRScoresSolver(
    implied::SemImplied,
    latent_vars::AbstractVector,
    A::AbstractMatrix,
    S::AbstractMatrix,
    lv_I_A⁻¹::Union{AbstractMatrix, Nothing} = nothing;
    alpha::Number = 0,
    prior_cov_alpha::Number = 0,
)
    ram = implied.ram_matrices
    lv_FA = Matrix(ram.F * A[:, latent_vars])

    prior_cov = if prior_cov_alpha != 0
        if isnothing(lv_I_A⁻¹)
            I_A = Matrix{eltype(A)}(I, size(A, 1), size(A, 2)) - A
            lv_I_A⁻¹ = inv_matrix_rows(I_A, latent_vars)
        end
        # postpone scaling by prior_cov_alpha until the Cholesky factor
        SEM.trunc_eigvals(
            Symmetric(X_A_Xt(S, lv_I_A⁻¹)),
            1e-6,
            mtx_label = "prior_cov",
            verbose = false,
        )
    else
        nothing
    end

    obs_inds = observed_var_indices(ram)
    obs_cov = Matrix(S[obs_inds, obs_inds])
    return QRScoresSolver(lv_FA, obs_cov; prior_cov, prior_cov_alpha, alpha)
end

latent_scores_solver(
    ::SemRegressionScores,
    implied::SemImplied,
    latent_vars::AbstractVector,
    A::AbstractMatrix,
    S::AbstractMatrix,
    lv_I_A⁻¹::Union{AbstractMatrix, Nothing} = nothing;
    alpha::Number = 0,
    prior_cov_alpha::Union{Number, Nothing} = nothing,
) = QRScoresSolver(
    implied,
    latent_vars,
    A,
    S,
    lv_I_A⁻¹;
    alpha,
    prior_cov_alpha = something(prior_cov_alpha, 1),
)

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

where `L_Ψ' L_Ψ = Ψ^{-1}`. Equivalently, this is the regression-score objective with
`prior_cov_alpha = 0`, i.e. with the latent-covariance prior switched off. Internally
the score operator is computed via the same augmented QR solve as the regression case.

# References

1. M. S. Bartlett, *The Statistical Conception of Mental Factors*, British Journal of
    Psychology, 28(1), 97-104, 1937.
2. Penn State STAT 505, *12.12 - Estimation of Factor Scores*:
    https://online.stat.psu.edu/stat505/lesson/12/12.12
3. UCLA Statistical Methods and Data Analytics, *A Practical Introduction to Factor Analysis*:
    https://stats.oarc.ucla.edu/spss/seminars/introduction-to-factor-analysis/
"""
struct SemBartlettScores <: SemScoresPredictMethod end

latent_scores_solver(
    ::SemBartlettScores,
    implied::SemImplied,
    latent_vars::AbstractVector,
    A::AbstractMatrix,
    S::AbstractMatrix,
    lv_I_A⁻¹::Union{AbstractMatrix, Nothing} = nothing;
    alpha::Number = 0,
    prior_cov_alpha::Nothing = nothing,
) = QRScoresSolver(
    implied,
    latent_vars,
    A,
    S,
    lv_I_A⁻¹;
    alpha,
    prior_cov_alpha = 0,
)

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

function latent_scores_solver(
    ::SemAndersonRubinScores,
    implied::SemImplied,
    latent_vars::AbstractVector,
    A::AbstractMatrix,
    S::AbstractMatrix,
    lv_I_A⁻¹::Union{AbstractMatrix, Nothing} = nothing;
    alpha::Number = 0,
    prior_cov_alpha::Nothing = nothing,
)
    nobs = nobserved_vars(implied)

    base_solver = QRScoresSolver(
        implied,
        latent_vars,
        A,
        S,
        lv_I_A⁻¹;
        alpha,
        prior_cov_alpha = 0,
    )
    base_op = permutedims(base_solver(Matrix{eltype(S)}(I, nobs, nobs)))

    score_cov = Symmetric(X_A_Xt(implied.Σ, base_op))
    return WhitenedScoresSolver(base_solver, cholesky!(score_cov))
end

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

"""
    score_basis_transform(model::SemLoss, params; method = :regression,
                          latent_vars = nothing, alpha = 0,
                          prior_cov_alpha = nothing)

Return the centered latent-score basis transform associated with a score-prediction
method.

For [`SemRegressionScores`](@ref) and [`SemBartlettScores`](@ref), the result is the
identity transform. For [`SemAndersonRubinScores`](@ref), the result stores the whitening
map between the model's original latent-variable basis and the Anderson-Rubin basis.

The returned object can be used to map centered Anderson-Rubin scores back to the
original latent-variable basis via:

```julia
orig_scores = transform(ar_scores; inverse = true)
```

`alpha` matters only for [`SemAndersonRubinScores`](@ref), because the whitening
transform is built from the ridge-regularized Bartlett score operator. For regression and
Bartlett scores the transform is the identity, so `alpha` does not affect the result.

`prior_cov_alpha` is currently accepted for API symmetry with
[`predict_latent_scores`](@ref) and for forward compatibility, but it does not affect the
returned transform for the currently implemented score methods.
"""
function score_basis_transform(
    method::SemScoresPredictMethod,
    model::SemLoss,
    params::AbstractVector;
    latent_vars::Union{AbstractVector, Nothing} = nothing,
    alpha::Number = 0,
    prior_cov_alpha::Union{Number, Nothing} = nothing,
)
    length(params) == nparams(model) || throw(
        DimensionMismatch(
            "The length of parameters vector ($(length(params))) does not match the number of parameters in the model ($(nparams(model))).",
        ),
    )
    alpha >= 0 ||
        throw(ArgumentError("The regularization parameter alpha must be non-negative"))
    isnothing(prior_cov_alpha) ||
        prior_cov_alpha >= 0 ||
        throw(
            ArgumentError(
                "The regularization parameter prior_cov_alpha must be non-negative",
            ),
        )

    implied = SEM.implied(model)
    ram = implied.ram_matrices
    lvar_inds =
        check_var_indices(ram, latent_vars, allow_observed = false, normalize = true)
    lvars = vars(ram)[lvar_inds]
    if !(method isa SemAndersonRubinScores) # identity transform
        return SemVariablesTransform(lvars, I, I)
    end

    update!(EvaluationTargets(0.0, nothing, nothing), implied, params)

    A = materialize(ram.A, params)
    S = materialize(ram.S, params)
    T = float(promote_type(eltype(A), eltype(S), typeof(alpha)))

    I_A = Matrix{eltype(A)}(I, size(A, 1), size(A, 2)) - A
    lv_I_A⁻¹ = inv_matrix_rows(I_A, lvar_inds)
    base_solver =
        QRScoresSolver(implied, lvar_inds, A, S, lv_I_A⁻¹; alpha, prior_cov_alpha = 0)
    nobs = nobserved_vars(implied)
    base_op = permutedims(base_solver(Matrix{T}(I, nobs, nobs)))
    score_cov = Symmetric(X_A_Xt(implied.Σ, base_op))
    score_cov_chol = cholesky!(score_cov)

    return SemVariablesTransform(lvars, I / score_cov_chol.U, score_cov_chol.U)
end

score_basis_transform(
    model::SemLoss,
    params::AbstractVector;
    method::Union{Symbol, SemScoresPredictMethod} = :regression,
    kwargs...
) = score_basis_transform(method, model, params; kwargs...)

score_basis_transform(
    method::Symbol,
    model::SemLoss,
    params::Union{AbstractVector, Nothing} = nothing;
    latent_vars::Union{AbstractVector, Nothing} = nothing,
    alpha::Number = 0,
    prior_cov_alpha::Union{Number, Nothing} = nothing,
) = score_basis_transform(
    SemScoresPredictMethod(method),
    model,
    params;
    latent_vars,
    alpha,
    prior_cov_alpha,
)

predict_latent_scores(
    fit::SemFit,
    data::SemObserved = observed(sem_term(fit.model));
    method::Symbol = :regression,
    latent_vars::Union{AbstractVector, Nothing} = nothing,
    alpha::Number = 0,
    prior_cov_alpha::Union{Number, Nothing} = nothing,
) = predict_latent_scores(
    SemScoresPredictMethod(method),
    fit,
    data;
    latent_vars,
    alpha,
    prior_cov_alpha,
)

predict_latent_scores(
    method::SemScoresPredictMethod,
    fit::SemFit,
    data::SemObserved = observed(sem_term(fit.model));
    latent_vars::Union{AbstractVector, Nothing} = nothing,
    alpha::Number = 0,
    prior_cov_alpha::Union{Number, Nothing} = nothing,
) = predict_latent_scores(
    method,
    loss(sem_term(fit.model)),
    fit.solution,
    data;
    latent_vars,
    alpha,
    prior_cov_alpha,
)

# return the rows of inv(A) for the specified row indices avoiding full inv(A) calculation
function inv_matrix_rows(A::AbstractMatrix, row_inds::AbstractVector{<:Integer})
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square."))
    rhs = zeros(eltype(A), n, length(row_inds))
    for (j, row_ix) in enumerate(row_inds)
        rhs[row_ix, j] = one(eltype(A))
    end
    return copy((A' \ rhs)')
end

"""
    predict_latent_scores(
        model::SemLoss, params, data = observed(model);
        method = :regression, latent_vars = nothing,
        alpha = 0, prior_cov_alpha = nothing
    )

Predict latent scores for the selected latent variables from observed data.

`method` selects the latent-score definition. See [`SemScoresPredictMethod`](@ref) and
its concrete implementations [`SemRegressionScores`](@ref),
[`SemBartlettScores`](@ref), and [`SemAndersonRubinScores`](@ref) for the mathematical
definitions and interpretation of each method.

`latent_vars` selects which latent variables are scored. If `latent_vars = nothing`, all
latent variables in the model are scored.

`alpha` is a non-negative ridge regularization parameter passed to the selected scoring method.
`prior_cov_alpha` is the non-negative weight of the latent-covariance prior used by
[`SemRegressionScores`](@ref). The default `prior_cov_alpha = 1` gives the classical
regression/Thomson scores, while `prior_cov_alpha = 0` reduces the regression objective
to the Bartlett objective with the same `alpha`.
"""
function predict_latent_scores(
    method::SemScoresPredictMethod,
    model::SemLoss,
    params::AbstractVector,
    data::SemObserved = observed(model);
    latent_vars::Union{AbstractVector, Nothing} = nothing,
    alpha::Number = 0,
    prior_cov_alpha::Union{Number, Nothing} = nothing,
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
    (alpha >= 0) ||
        throw(ArgumentError("The regularization parameter alpha must be non-negative"))
    if method isa Union{SemBartlettScores, SemAndersonRubinScores}
        !isnothing(prior_cov_alpha)
        @warn "prior_cov_alpha is only supported for regression scores, ignored for $(typeof(method))"
        prior_cov_alpha = nothing
    end
    isnothing(prior_cov_alpha) ||
        prior_cov_alpha >= 0 ||
        throw(
            ArgumentError(
                "The regularization parameter prior_cov_alpha must be non-negative",
            ),
        )

    implied = SEM.implied(model)
    ram = implied.ram_matrices
    lv_inds = check_var_indices(ram, latent_vars, allow_observed = false, normalize = true)

    update!(EvaluationTargets(0.0, nothing, nothing), implied, params)

    A = materialize(ram.A, params)
    S = materialize(ram.S, params)
    I_A = Matrix{eltype(A)}(I, size(A, 1), size(A, 2)) - A
    lv_I_A⁻¹ = inv_matrix_rows(I_A, lv_inds)
    lv_scores_solver = latent_scores_solver(
        method,
        implied,
        lv_inds,
        A,
        S,
        lv_I_A⁻¹;
        alpha,
        prior_cov_alpha,
    )

    centered_data =
        data.data .- (isnothing(data.obs_mean) ? mean(data.data, dims = 1) : data.obs_mean')
    lv_scores = lv_scores_solver(centered_data)

    # adjust the scores w.r.t the variable means
    if MeanStruct(implied) === HasMeanStruct
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
    latent_vars::Union{AbstractVector, Nothing} = nothing,
    alpha::Number = 0,
    prior_cov_alpha::Union{Number, Nothing} = nothing,
) = predict_latent_scores(
    SemScoresPredictMethod(method),
    model,
    params,
    data;
    latent_vars,
    alpha,
    prior_cov_alpha,
)
