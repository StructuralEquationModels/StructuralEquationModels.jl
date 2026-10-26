# generic penalties on transformed parameter vectors

############################################################################################
### Types
############################################################################################

"""
    struct SemParamsPenalty{H, TP, TA, TB, TS1, TS2, TW2, TT} <: AbstractLoss

Generic regularization term acting on SEM parameters, optionally after an affine
transform `A * p + b`.

Without a hinge bound (`bound = :none`), the elementwise penalty is
``|z_i|^{s_1} + w_2 * |z_i|^{s_2}``. The second term is optional.
With `bound = :l` or `bound = :u`, the same powers are applied to the
corresponding one-sided hinge-transformed z instead. This single type backs `SemNorm`,
`SemLasso`, `SemRidge`, and `SemHinge`.

# Constructors
    SemParamsPenalty([params], [A], [b]; shape1 = 1, shape2 = nothing,
                     weight2 = nothing,
                     threshold = 0.0, bound = :none)
    SemParamsPenalty(spec::SemSpecification, params::AbstractVector,
                     [A::AbstractMatrix = nothing],
                     [b::AbstractVector = nothing];
                     shape1 = 1, shape2 = nothing,
                     weight2 = nothing,
                     threshold = 0.0, bound = :none)

# See also

- Wrappers for `SemParamsPenalty` providing default regularization terms:
  [`SemNorm`](@ref), [`SemLasso`](@ref), [`SemRidge`](@ref), [`SemElasticNet`](@ref), and
  [`SemHinge`](@ref).
"""
struct SemParamsPenalty{H, TP, TA, TB, TS1, TS2, TW2, TT} <: AbstractLoss
    hessianeval::ExactHessian
    param_inds::TP
    A::TA
    At::TA
    b::TB
    shape1::TS1
    shape2::TS2
    weight2::TW2
    threshold::TT
end

############################################################################
### Constructors
############################################################################

function SemParamsPenalty(
    param_inds::Union{AbstractVector, Nothing} = nothing,
    A::Union{AbstractMatrix, Nothing} = nothing,
    b::Union{AbstractVector, Nothing} = nothing;
    shape1::Real = 1,
    shape2::Union{Real, Nothing} = nothing,
    weight2::Union{Real, Nothing} = nothing,
    threshold::Real = 0.0,
    bound::Symbol = :none,
)
    bound ∈ (:none, :l, :u) ||
        throw(ArgumentError("bound must be :none, :l, or :u, $bound given"))

    shape1 > 0 || throw(ArgumentError("shape1 must be positive, got $shape1"))
    isnothing(shape2) || shape2 > 0 ||
        throw(ArgumentError("shape2 must be positive when provided, got $shape2"))

    isnothing(param_inds) || !(eltype(param_inds) <: Symbol) ||
        throw(ArgumentError("Symbol parameter ids require a SemSpecification."))

    isnothing(A) || isnothing(param_inds) || size(A, 2) == length(param_inds) ||
        throw(
            DimensionMismatch(
                "The transformation matrix columns ($(size(A, 2))) should match " *
                "the number of parameters to regularize ($(length(param_inds)))",
            ),
        )

    if !isnothing(b)
        (!isnothing(A) || !isnothing(param_inds)) ||
            throw(
                ArgumentError(
                    "An intercept vector requires either a parameter subset or a transformation matrix.",
                ),
            )

        nused_params = isnothing(param_inds) ? size(A, 2) : length(param_inds)
        npenalty_els = isnothing(A) ? length(param_inds) : size(A, 1)
        length(b) == npenalty_els ||
            throw(
                DimensionMismatch(
                    "The intercept length ($(length(b))) should match " *
                    (!isnothing(A) ? "the rows of the transformation matrix" :
                    "the number of parameters to regularize") *
                    " ($nused_params)",
                ),
            )
    end

    if isnothing(shape2)
        isnothing(weight2) ||
            throw(ArgumentError("weight2 can only be specified when shape2 is specified."))
        weight2 = nothing
    else
        isnothing(weight2) &&
            throw(ArgumentError("weight2 must be specified when shape2 is specified."))
    end

    At = !isnothing(A) ? permutedims(A) : nothing
    shape1 = isone(shape1) ? 1 : shape1 == 2 ? 2 : shape1
    shape2 = isnothing(shape2) ? nothing : (isone(shape2) ? 1 : shape2 == 2 ? 2 : shape2)
    threshold_value = float(threshold)

    return SemParamsPenalty{
        bound,
        typeof(param_inds),
        typeof(A),
        typeof(b),
        typeof(shape1),
        typeof(shape2),
        typeof(weight2),
        typeof(threshold_value),
    }(
        ExactHessian(),
        param_inds,
        A,
        At,
        b,
        shape1,
        shape2,
        weight2,
        threshold_value,
    )
end

function SemParamsPenalty(
    spec::SemSpecification,
    params::AbstractVector,
    A::Union{AbstractMatrix, Nothing} = nothing,
    b::Union{AbstractVector, Nothing} = nothing;
    shape1::Real = 1,
    shape2::Union{Real, Nothing} = nothing,
    weight2::Union{Real, Nothing} = nothing,
    threshold::Real = 0.0,
    bound::Symbol = :none,
)
    param_inds = eltype(params) <: Symbol ? param_indices(spec, params) : params

    isnothing(A) ||
        size(A, 2) == length(param_inds) ||
        throw(
            DimensionMismatch(
                "The transformation matrix columns ($(size(A, 2))) should match " *
                "the parameters to regularize ($(length(param_inds)))",
            ),
        )

    sel_params_mtx = eachrow_to_col(Float64, param_inds, nparams(spec))
    if !isnothing(A)
        if A isa SparseMatrixCSC
            # for sparse case, combine the matrix transform and
            # parameter selection into a single matrix
            A = convert(typeof(A), A * sel_params_mtx)
            param_inds = nothing
        end
    else
        A = sel_params_mtx
        param_inds = nothing
    end

    return SemParamsPenalty(
        param_inds,
        A,
        b;
        shape1,
        shape2,
        weight2,
        threshold,
        bound,
    )
end

SemNorm(args...; shape::Real = 1) = SemParamsPenalty(args...; shape1 = shape)
SemLasso(args...; kwargs...) = SemNorm(args...; shape = 1, kwargs...)
SemRidge(args...; kwargs...) = SemNorm(args...; shape = 2, kwargs...)
SemElasticNet(args...; shape1::Real = 1, shape2::Real = 2, weight2::Real = 1) =
    SemParamsPenalty(args...; shape1, shape2, weight2)

SemHinge(
    params::AbstractVector,
    A::Union{AbstractMatrix, Nothing} = nothing,
    b::Union{AbstractVector, Nothing} = nothing;
    bound::Symbol = :l,
    threshold::Number = 0.0,
    shape1::Real = 1,
    shape2::Union{Real, Nothing} = nothing,
    weight2::Union{Real, Nothing} = nothing,
) = SemParamsPenalty(params, A, b; shape1, shape2, weight2, threshold, bound)

SemHinge(
    spec::SemSpecification,
    params::AbstractVector,
    A::Union{AbstractMatrix, Nothing} = nothing,
    b::Union{AbstractVector, Nothing} = nothing;
    bound::Symbol = :l,
    threshold::Number = 0.0,
    shape1::Real = 1,
    shape2::Union{Real, Nothing} = nothing,
    weight2::Union{Real, Nothing} = nothing,
) = SemParamsPenalty(spec, params, A, b; shape1, shape2, weight2, threshold, bound)

nparams(f::SemParamsPenalty) = !isnothing(f.A) ? size(f.A, 2) : (isnothing(f.param_inds) ? 0 : length(f.param_inds))

############################################################################################
### methods
############################################################################################

_elhinge(x, ::Val{:none}, threshold) = abs(x)
_elhinge(x, ::Val{:l}, threshold) = max(x - threshold, zero(x))
_elhinge(x, ::Val{:u}, threshold) = max(threshold - x, zero(x))

_elhinge_direction(x, ::Val{:none}, threshold) = sign(x)
_elhinge_direction(x, ::Val{:l}, threshold) = x > threshold ? one(x) : zero(x)
_elhinge_direction(x, ::Val{:u}, threshold) = x < threshold ? -one(x) : zero(x)

_elhinge_isactive(x, ::Val{:none}, threshold) = true
_elhinge_isactive(x, ::Val{:l}, threshold) = x > threshold
_elhinge_isactive(x, ::Val{:u}, threshold) = x < threshold

elnorm(x, ::Val{1}, mode, threshold) = _elhinge(x, mode, threshold)
elnorm(x, ::Val{2}, mode, threshold) = abs2(_elhinge(x, mode, threshold))
elnorm(x, ::Val{S}, mode, threshold) where {S} = _elhinge(x, mode, threshold)^S

elnormgrad(x, ::Val{1}, mode, threshold) = _elhinge_direction(x, mode, threshold)
elnormgrad(x, ::Val{2}, mode, threshold) =
    (one(x) + one(x)) * _elhinge(x, mode, threshold) * _elhinge_direction(x, mode, threshold)

elnormgrad(x, ::Val{S}, mode, threshold) where {S} =
    _elhinge_isactive(x, mode, threshold) ?
        S * _elhinge(x, mode, threshold)^(S - 1) * _elhinge_direction(x, mode, threshold) :
        zero(x)

elnormhdiag(x, ::Val{1}, mode, threshold) = zero(x)
elnormhdiag(x, ::Val{2}, mode, threshold) = _elhinge_isactive(x, mode, threshold) ? (one(x) + one(x)) : zero(x)

elnormhdiag(x, ::Val{S}, mode, threshold) where {S} =
    _elhinge_isactive(x, mode, threshold) ?
        S * (S - 1) * _elhinge(x, mode, threshold)^(S - 2) :
        zero(x)

Base.@propagate_inbounds @inline function _elevaluate!(objective, elm_grad, elm_hdiag, x, i, weight, shape, mode, threshold)
    isnothing(objective) || (objective += weight * elnorm(x, shape, mode, threshold))
    isnothing(elm_grad) || (elm_grad[i] += weight * elnormgrad(x, shape, mode, threshold))
    isnothing(elm_hdiag) || (elm_hdiag[i] += weight * elnormhdiag(x, shape, mode, threshold))

    return objective
end

Base.@propagate_inbounds @inline function _elevaluate!(objective, elm_grad, elm_hdiag, x, i, weight::Nothing, shape, mode, threshold)
    isnothing(objective) || (objective += elnorm(x, shape, mode, threshold))
    isnothing(elm_grad) || (elm_grad[i] += elnormgrad(x, shape, mode, threshold))
    isnothing(elm_hdiag) || (elm_hdiag[i] += elnormhdiag(x, shape, mode, threshold))

    return objective
end

function _evaluate!(
    objective,
    gradient,
    hessian,
    penalty::SemParamsPenalty,
    params,
    shape1::Val{S1},
    shape2::Val{S2},
    mode::Val{H},
) where {S1, S2, H}
    sel_params = isnothing(penalty.param_inds) ? params : params[penalty.param_inds]
    trf_params = isnothing(penalty.A) ?
        sel_params :
        penalty.A * sel_params
    if !isnothing(penalty.b)
        trf_params === params && (trf_params = copy(trf_params))
        trf_params .+= penalty.b
    end

    weight2 = isnothing(penalty.weight2) ? zero(eltype(trf_params)) : penalty.weight2
    T = promote_type(eltype(trf_params), typeof(weight2))

    obj = isnothing(objective) ? nothing : zero(T)
    elm_grad = isnothing(gradient) ? nothing : fill!(similar(trf_params, T), zero(T))
    elm_hdiag = isnothing(hessian) ? nothing : fill!(similar(trf_params, T), zero(T))

    @inbounds for i in eachindex(trf_params)
         obj = _elevaluate!(obj, elm_grad, elm_hdiag, trf_params[i], i, nothing, shape1, mode, penalty.threshold)
    end

    if !isnothing(S2)
        @inbounds for i in eachindex(trf_params)
            obj = _elevaluate!(obj, elm_grad, elm_hdiag, trf_params[i], i, weight2, shape2, mode, penalty.threshold)
        end
    end

    if !isnothing(gradient)
        if isnothing(penalty.A)
            if isnothing(penalty.param_inds)
                copyto!(gradient, elm_grad)
            else
                fill!(gradient, zero(eltype(gradient)))
                @inbounds gradient[penalty.param_inds] .= elm_grad
            end
        elseif isnothing(penalty.param_inds)
            mul!(gradient, penalty.At, elm_grad)
        else
            local_grad = penalty.A' * elm_grad
            fill!(gradient, 0)
            @inbounds gradient[penalty.param_inds] .= local_grad
        end
    end

    if !isnothing(hessian)
        fill!(hessian, 0)
        if !all(iszero, elm_hdiag)
            @inbounds if isnothing(penalty.A)
                if isnothing(penalty.param_inds)
                    hessian[diagind(hessian)] .= elm_hdiag
                else
                    hessian[diagind(hessian)[penalty.param_inds]] .= elm_hdiag
                end
            else
                local_hessian = penalty.A' * Diagonal(elm_hdiag) * penalty.A
                if isnothing(penalty.param_inds)
                    copyto!(hessian, local_hessian)
                else
                    hessian[penalty.param_inds, penalty.param_inds] .= local_hessian
                end
            end
        end
    end

    return isnothing(objective) ? NaN : obj
end

evaluate!(objective, gradient, hessian, penalty::SemParamsPenalty{H}, params) where {H} =
    _evaluate!(
        objective,
        gradient,
        hessian,
        penalty,
        params,
        Val(penalty.shape1),
        Val(penalty.shape2),
        Val(H),
    )
