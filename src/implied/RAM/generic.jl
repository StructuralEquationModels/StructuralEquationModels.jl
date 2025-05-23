############################################################################################
### Types
############################################################################################
@doc raw"""
Model implied covariance and means via RAM notation.

# Constructor

    RAM(;specification,
        meanstructure = false,
        gradient = true,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object
- `meanstructure::Bool`: does the model have a meanstructure?
- `gradient::Bool`: is gradient-based optimization used

# Extended help

## RAM notation

The model implied covariance matrix is computed as
```math
    \Sigma = F(I-A)^{-1}S(I-A)^{-T}F^T
```
and for models with a meanstructure, the model implied means are computed as
```math
    \mu = F(I-A)^{-1}M
```

## Interfaces
- `param_labels(::RAM) `-> vector of parameter labels
- `nparams(::RAM)` -> number of parameters

- `ram.Σ` -> model implied covariance matrix
- `ram.μ` -> model implied mean vector

RAM matrices for the current parameter values:
- `ram.A`
- `ram.S`
- `ram.F`
- `ram.M`

Jacobians of RAM matrices w.r.t to the parameter vector `θ`
- `ram.∇A` -> ``∂vec(A)/∂θᵀ``
- `ram.∇S` -> ``∂vec(S)/∂θᵀ``
- `ram.∇M` = ``∂M/∂θᵀ``

Vector of indices of each parameter in the respective RAM matrix:
- `ram.A_indices`
- `ram.S_indices`
- `ram.M_indices`

Additional interfaces
- `ram.F⨉I_A⁻¹` -> ``F(I-A)^{-1}``
- `ram.F⨉I_A⁻¹S` -> ``F(I-A)^{-1}S``
- `ram.I_A` -> ``I-A``

Only available in gradient! calls:
- `ram.I_A⁻¹` -> ``(I-A)^{-1}``
"""
mutable struct RAM{MS, A1, A2, A3, A4, A5, A6, V2, M1, M2, M3, M4, S1, S2, S3} <: SemImplied
    meanstruct::MS
    hessianeval::ExactHessian

    Σ::A1
    A::A2
    S::A3
    F::A4
    μ::A5
    M::A6

    ram_matrices::V2

    F⨉I_A⁻¹::M1
    F⨉I_A⁻¹S::M2
    I_A::M3
    I_A⁻¹::M4

    ∇A::S1
    ∇S::S2
    ∇M::S3

    RAM{MS}(args...) where {MS <: MeanStruct} =
        new{MS, map(typeof, args)...}(MS(), ExactHessian(), args...)
end

############################################################################################
### Constructors
############################################################################################

function RAM(;
    specification::SemSpecification,
    gradient_required = true,
    meanstructure = false,
    kwargs...,
)
    ram_matrices = convert(RAMMatrices, specification)

    # get dimensions of the model
    n_par = nparams(ram_matrices)
    n_obs = nobserved_vars(ram_matrices)
    n_var = nvars(ram_matrices)

    #preallocate arrays
    rand_params = randn(Float64, n_par)
    A_pre = check_acyclic(materialize(ram_matrices.A, rand_params))
    S_pre = materialize(ram_matrices.S, rand_params)
    F = copy(ram_matrices.F)

    # pre-allocate some matrices
    Σ = zeros(n_obs, n_obs)
    F⨉I_A⁻¹ = zeros(n_obs, n_var)
    F⨉I_A⁻¹S = zeros(n_obs, n_var)
    I_A = similar(A_pre)

    if gradient_required
        ∇A = sparse_gradient(ram_matrices.A)
        ∇S = sparse_gradient(ram_matrices.S)
    else
        ∇A = nothing
        ∇S = nothing
    end

    # μ
    if meanstructure
        MS = HasMeanStruct
        !isnothing(ram_matrices.M) || throw(
            ArgumentError(
                "You set `meanstructure = true`, but your model specification contains no mean parameters.",
            ),
        )
        M_pre = materialize(ram_matrices.M, rand_params)
        ∇M = gradient_required ? sparse_gradient(ram_matrices.M) : nothing
        μ = zeros(n_obs)
    else
        MS = NoMeanStruct
        M_pre = nothing
        μ = nothing
        ∇M = nothing
    end

    return RAM{MS}(
        Σ,
        A_pre,
        S_pre,
        F,
        μ,
        M_pre,
        ram_matrices,
        F⨉I_A⁻¹,
        F⨉I_A⁻¹S,
        I_A,
        copy(I_A),
        ∇A,
        ∇S,
        ∇M,
    )
end

############################################################################################
### methods
############################################################################################

function update!(targets::EvaluationTargets, implied::RAM, model::AbstractSemSingle, param_labels)
    materialize!(implied.A, implied.ram_matrices.A, param_labels)
    materialize!(implied.S, implied.ram_matrices.S, param_labels)
    if !isnothing(implied.M)
        materialize!(implied.M, implied.ram_matrices.M, param_labels)
    end

    parent(implied.I_A) .= .-implied.A
    @view(implied.I_A[diagind(implied.I_A)]) .+= 1

    if is_gradient_required(targets) || is_hessian_required(targets)
        implied.I_A⁻¹ = LinearAlgebra.inv!(factorize(implied.I_A))
        mul!(implied.F⨉I_A⁻¹, implied.F, implied.I_A⁻¹)
    else
        copyto!(implied.F⨉I_A⁻¹, implied.F)
        rdiv!(implied.F⨉I_A⁻¹, factorize(implied.I_A))
    end

    mul!(implied.F⨉I_A⁻¹S, implied.F⨉I_A⁻¹, implied.S)
    mul!(parent(implied.Σ), implied.F⨉I_A⁻¹S, implied.F⨉I_A⁻¹')

    if MeanStruct(implied) === HasMeanStruct
        mul!(implied.μ, implied.F⨉I_A⁻¹, implied.M)
    end
end

############################################################################################
### Recommended methods
############################################################################################

function update_observed(implied::RAM, observed::SemObserved; kwargs...)
    if nobserved_vars(observed) == size(implied.Σ, 1)
        return implied
    else
        return RAM(; observed = observed, kwargs...)
    end
end
