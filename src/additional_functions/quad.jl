_unwrap_symmetric(res::AbstractMatrix) = res
_unwrap_symmetric(res::Symmetric) = parent(res)

# faster version of copytri!() that uses blascopy!()
function blascopytri!(A::StridedMatrix, uplo::AbstractChar)
    n = LinearAlgebra.checksquare(A)
    if uplo == 'L'
        for (i, di) in enumerate(diagind(A))
            (i < n) || continue
            BLAS.blascopy!(
                n - i,
                pointer(A, di + 1),
                stride(A, 1),
                pointer(A, di + size(A, 2)),
                stride(A, 2),
            )
        end
    elseif uplo == 'U'
        for (i, di) in enumerate(diagind(A))
            (i < n) || continue
            BLAS.blascopy!(
                n - i,
                pointer(A, di + size(A, 2)),
                stride(A, 2),
                pointer(A, di + 1),
                stride(A, 1),
            )
        end
    else
        lazy"uplo argument must be 'U' (upper) or 'L' (lower), got $uplo" |>
        ArgumentError |>
        throw
    end
    return A
end

# faster copytri!() that uses @simd, @inbounds and drops elementwise conjugation
@inline function fastcopytri!(A::AbstractMatrix, uplo::AbstractChar)
    n = LinearAlgebra.checksquare(A)
    if uplo == 'U'
        @inbounds for i in axes(A, 1)
            @simd for j in (i+1):n
                A[j, i] = A[i, j]
            end
        end
    elseif uplo == 'L'
        @inbounds for i in axes(A, 1)
            @simd for j in (i+1):n
                A[i, j] = A[j, i]
            end
        end
    else
        lazy"uplo argument must be 'U' (upper) or 'L' (lower), got $uplo" |>
        ArgumentError |>
        throw
    end
    A
end

# faster version that drops issymmetric checks
# and switches to gemm mode for large matrices
@inline function syrk_wrapper!(
    res::AbstractMatrix,
    trans::Char,
    X::Union{StridedMatrix, StridedVector},
    alpha::Real = 1,
    beta::Real = 0;
    check::Bool = true,
    # big matrices are multiplied in gemm mode to avoid long copytri!()
    mode::Symbol = size(res, 1) >= 1000 ? :gemm : :syrk,
)
    T = eltype(X)
    if mode == :syrk && (iszero(beta) || (!check || issymmetric(res)))
        BLAS.syrk!('U', trans, T(alpha), X, T(beta), _unwrap_symmetric(res))
        fastcopytri!(_unwrap_symmetric(res), 'U')
    elseif mode == :gemm # generic
        LinearAlgebra.gemm_wrapper!(
            _unwrap_symmetric(res),
            trans,
            trans == 'N' ? 'T' : 'N',
            X,
            X,
            LinearAlgebra.MulAddMul(alpha, beta),
        )
    else
        throw(ArgumentError(lazy"mode must be :syrk or :gemm, $mode given"))
    end
    return res
end

# calculate Xᵀ⋅X
Xt_X!(res::AbstractMatrix, X::AbstractMatrix, alpha::Real = 1, beta::Real = 0) =
    mul!(_unwrap_symmetric(res), X', X, alpha, beta)

Xt_X!(res::AbstractMatrix, X::StridedMatrix, alpha::Real = 1, beta::Real = 0; kwargs...) =
    syrk_wrapper!(res, 'T', X, alpha, beta; kwargs...)

X_Xt!(
    res::AbstractMatrix,
    X::Union{AbstractMatrix, AbstractVector},
    alpha::Real = 1,
    beta::Real = 0,
) = mul!(_unwrap_symmetric(res), X, X', alpha, beta)

X_Xt!(
    res::AbstractMatrix,
    X::Union{StridedMatrix, StridedVector},
    alpha::Real = 1,
    beta::Real = 0;
    kwargs...,
) = syrk_wrapper!(res, 'N', X, alpha, beta; kwargs...)

Xt_X(X::AbstractMatrix) = Xt_X!(Matrix{eltype(X)}(undef, size(X, 2), size(X, 2)), X)

X_Xt(X::Union{AbstractMatrix, AbstractVector}) =
    X_Xt!(Matrix{eltype(X)}(undef, size(X, 1), size(X, 1)), X)

# calculate Xᵀ⋅A⋅X
# FIXME: use PDMats.jl when its sparse matrix support is refactored
# see https://github.com/JuliaStats/PDMats.jl/pull/188
function Xt_A_X!(
    res::AbstractMatrix,
    A::AbstractMatrix,
    X::AbstractMatrix,
    alpha::Real = 1,
    beta::Real = 0;
    Xt_A_buf::Union{AbstractMatrix, Nothing} = nothing,
)
    Xt_A = !isnothing(Xt_A_buf) ? mul!(Xt_A_buf, X', A) : X'A
    return mul!(_unwrap_symmetric(res), Xt_A, X, alpha, beta)
end

# special handling of symmetric to make sure it is the first argument in *
function Xt_A_X!(
    res::AbstractMatrix,
    A::Symmetric{<:Any, M},
    X::AbstractMatrix,
    alpha::Real = 1,
    beta::Real = 0;
    Xt_A_buf::Union{AbstractMatrix, Nothing} = nothing,
) where {M <: StridedMatrix}
    A_X = !isnothing(Xt_A_buf) ? mul!(reshape(Xt_A_buf, size(X)), A, X) : A*X
    return mul!(_unwrap_symmetric(res), X', A_X, alpha, beta)
end

Xt_A_X(A::AbstractMatrix, X::AbstractMatrix, alpha::Real = 1; kwargs...) = Xt_A_X!(
    Matrix{promote_type(eltype(A), eltype(X))}(undef, size(X, 2), size(X, 2)),
    A,
    X,
    alpha,
    0;
    kwargs...,
)

function X_A_Xt!(
    res::AbstractMatrix,
    A::AbstractMatrix,
    X::AbstractMatrix,
    alpha::Real = 1,
    beta::Real = 0;
    X_A_buf::Union{AbstractMatrix, Nothing} = nothing,
)
    X_A = !isnothing(X_A_buf) ? mul!(X_A_buf, X, A) : X * A
    return mul!(_unwrap_symmetric(res), X_A, X', alpha, beta)
end

# special handling of symmetric to make sure it is the first argument in *
function X_A_Xt!(
    res::AbstractMatrix,
    A::Symmetric{<:Any, M},
    X::AbstractMatrix,
    alpha::Real = 1,
    beta::Real = 0;
    X_A_buf::Union{AbstractMatrix, Nothing} = nothing,
) where {M <: StridedMatrix}
    # FIXME in principle no need to unwrap A, but with symmetric A and transposed X
    # julia's generic_matmatmul() falls back into non-BLAS implementation (looks like Julia's bug)
    A_Xt =
        !isnothing(X_A_buf) ?
        mul!(reshape(X_A_buf, size(X, 2), size(X, 1)), _unwrap_symmetric(A), X') :
        _unwrap_symmetric(A) * X'
    return mul!(_unwrap_symmetric(res), X, A_Xt, alpha, beta)
end

X_A_Xt(A::AbstractMatrix, X::AbstractMatrix, alpha::Real = 1; kwargs...) = X_A_Xt!(
    Matrix{promote_type(eltype(A), eltype(X))}(undef, size(X, 1), size(X, 1)),
    A,
    X,
    alpha,
    0;
    kwargs...,
)
