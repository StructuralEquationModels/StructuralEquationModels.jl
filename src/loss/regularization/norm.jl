# l^α regularization

############################################################################################
### Types
############################################################################################
"""
    struct SemNorm{α, T, TB} <: AbstractLoss{ExactHessian}

Regularization term that provides *Lᵅ* regularization of SEM parameters.
The term implements the ``\\sum_{i=1\\ldots n} \\left| p_i \right|^{\\alpha}``,
where *p_i*, *i = 1..n* is the vector of selected SEM parameter values.
For `α = 1` it implements the *LASSO* (`SemLasso` alias type), and for
`α = 2` it implements the *Ridge* regularization (`SemRidge` alias type).
The term also allows specifying an optional affine transform (*A × p + b*)
to apply to the parameters before the regularization.

# Constructors
    SemNorm(A::SparseMatrixCSC, b::Union{AbstractVector, Nothing} = nothing)
    SemNorm(spec::SemSpecification, params::AbstractVector,
            [A::AbstractMatrix = nothing],
            [b::AbstractVector = nothing];
            α::Real)

# Arguments
- `spec`: SEM model specification.
- `params::Vector`: optional IDs (Symbols) or indices of parameters to regularize.
- `A`: optional transformation matrix that defines how to transform the vector of parameter values
       before the regularization. If `params` is not specified, the transformation is applied
       to the entire parameters vector.
- `b`: optional vector of intercepts to add to the transformed parameters.
- `α`: regularization parameter, any positive real number is supported

# Examples
```julia
my_lasso = SemLasso(spec, [:λ₁, :λ₂, :ω₂₃])
my_trans_ridge = SemRidge(spec, [:λ₁, :λ₂, :ω₂₃], [1.0 1.0 0.0; 0.0 0.0 1.0], [-2.0, 0.0])
```
"""
struct SemNorm{α, TP, TA, TB, TH} <: AbstractLoss{ExactHessian}
    param_inds::TP          # indices of parameters to regularize
    A::TA                   # transformation/subsetting of the parameters
    At::TA                  # Aᵀ
    b::TB                   # optional transformed parameter intercepts
    H_inds::Vector{Int}     # non-zero linear indices of Hessian
    H_vals::TH              # non-zero values of Hessian
end

const SemRidge{TP, TA, TB, TH} = SemNorm{2, TP, TA, TB, TH}
const SemLasso{TP, TA, TB, TH} = SemNorm{1, TP, TA, TB, TH}

############################################################################
### Constructors
############################################################################

function SemNorm{α}(
    param_inds::Union{AbstractVector, Nothing} = nothing,
    A::Union{AbstractMatrix, Nothing} = nothing,
    b::Union{AbstractVector, Nothing} = nothing,
) where {α}
    isnothing(A) || isnothing(param_inds) || size(A, 2) == length(param_inds) ||
        throw(DimensionMismatch("The transformation matrix columns ($(size(A, 2))) should match " *
                                "the number of parameters to regularize ($(length(param_inds)))"))
    isnothing(b) || (isnothing(A) && isnothing(param_inds)) ||
    (length(b) == (isnothing(A) ? length(param_inds) : size(A, 1))) ||
        throw(DimensionMismatch("The intercept length ($(length(b))) should match the rows of " *
                                "the transformation matrix ($(isnothing(A) ? "not specified" : size(A, 1)))" *
                                " or the number of parameters to regularize ($(isnothing(param_inds) ? "not specified" : length(param_inds)))"))

    At = !isnothing(A) ? convert(typeof(A), A') : nothing
    H = !isnothing(A) ? α * At * A : nothing # FIXME
    if isnothing(H)
        H_inds = Vector{Int}()
        H_v = nothing
    else
        H_i, H_j, H_v = findnz(H)
        H_indmtx = LinearIndices(H)
        H_inds = [H_indmtx[i, j] for (i, j) in zip(H_i, H_j)]
        H_v = copy(H_v)
    end
    return SemNorm{α, typeof(param_inds), typeof(A), typeof(b), typeof(H_v)}(
                param_inds, A, At, b, H_inds, H_v)
end

function SemNorm{α}(
    spec::SemSpecification,
    params::AbstractVector,
    A::Union{AbstractMatrix, Nothing} = nothing,
    b::Union{AbstractVector, Nothing} = nothing,
) where {α}
    param_inds = eltype(params) <: Symbol ? param_indices(spec, params) : params

    isnothing(A) || size(A, 2) == length(param_inds) ||
        throw(DimensionMismatch("The transformation matrix columns ($(size(A, 2))) should match " *
                                "the parameters to regularize ($(length(param_inds)))"))

    sel_params_mtx = eachrow_to_col(Float64, param_inds, nparams(spec))
    if !isnothing(A)
        if A isa SparseMatrixCSC
            # for sparse matrices do parameters selection and multiplication in one step
            A = convert(typeof(A), A * sel_params_mtx)
            param_inds = nothing
        end
    else # if no matrix, just use selection matrix
        A = sel_params_mtx
        param_inds = nothing
    end
    return SemNorm{α}(param_inds, A, b)
end

SemNorm(args...; α::Real) = SemNorm{α}(args...)

nparams(f::SemNorm) = size(f.A, 2)

############################################################################################
### methods
############################################################################################

elnorm(_::Val{α}) where {α} = x -> abs(x)^α
elnorm(_::Val{1}) = abs
elnorm(_::Val{2}) = abs2

elnorm(_::SemNorm{α}) where {α} = elnorm(Val(α))

# not multiplied by α, handled by mul!
elnormgrad(_::Val{α}) where {α} = x -> abs(x)^(α - 1) * sign(x)
elnormgrad(_::Val{2}) = identity
elnormgrad(_::Val{1}) = sign

elnormgrad(::SemNorm{α}) where {α} = elnormgrad(Val(α))

function evaluate!(
    objective, gradient, hessian,
    norm::SemNorm{α},
    params,
) where {α}
    if !isnothing(norm.param_inds)
        params = params[norm.param_inds]
    end
    if !isnothing(norm.A)
        trf_params = norm.A * params
    end
    if !isnothing(norm.b)
        trf_params .+= norm.b
    end

    obj = NaN
    isnothing(objective) || (obj = sum(elnorm(norm), trf_params))

    if !isnothing(gradient)
        elgrad_trf_params = elnormgrad(norm).(trf_params)
        if !isnothing(norm.param_inds)
            mul!(params, norm.At, elgrad_trf_params, α, 0)
            fill!(gradient, 0)
            @inbounds gradient[norm.param_inds] .= params
        else
            mul!(gradient, norm.At, elgrad_trf_params, α, 0)
        end
    end

    if !isnothing(hessian)
        fill!(hessian, 0)
        if α === 1
            # do nothing, hessian is zero
        elseif α === 2
            @inbounds hessian[norm.H_inds] .= norm.H_vals
        else
            error("Hessian not implemented for α ≠ 1, 2")
            # TODO: Implement Hessian for other values of α
        end
    end
    return obj
end
