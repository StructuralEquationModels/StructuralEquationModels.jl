############################################################################################
### Types
############################################################################################
@doc raw"""
Model implied covariance and means via RAM notation.

# Constructor

    RAM(;
        specification,
        meanstructure = false,
        gradient = true,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object
- `meanstructure::Bool`: does the model have a meanstructure?
- `gradient::Bool`: is gradient-based optimization used

# Extended help

## Implementation
Subtype of `SemImply`.

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
- `params(::RAM) `-> vector of parameter labels
- `nparams(::RAM)` -> number of parameters

- `Σ(::RAM)` -> model implied covariance matrix
- `μ(::RAM)` -> model implied mean vector

RAM matrices for the current parameter values:
- `A(::RAM)`
- `S(::RAM)`
- `F(::RAM)`
- `M(::RAM)`

Jacobians of RAM matrices w.r.t to the parameter vector `θ`
- `∇A(::RAM)` -> ``∂vec(A)/∂θᵀ``
- `∇S(::RAM)` -> ``∂vec(S)/∂θᵀ``
- `∇M(::RAM)` = ``∂M/∂θᵀ``

Vector of indices of each parameter in the respective RAM matrix:
- `A_indices(::RAM)`
- `S_indices(::RAM)`
- `M_indices(::RAM)`

Additional interfaces
- `F⨉I_A⁻¹(::RAM)` -> ``F(I-A)^{-1}``
- `F⨉I_A⁻¹S(::RAM)` -> ``F(I-A)^{-1}S``
- `I_A(::RAM)` -> ``I-A``
- `has_meanstructure(::RAM)` -> `Val{Bool}` does the model have a meanstructure?

Only available in gradient! calls:
- `I_A⁻¹(::RAM)` -> ``(I-A)^{-1}``
"""
mutable struct RAM{A1, A2, A3, A4, A5, A6, V2, I1, I2, I3, M1, M2, M3, M4, S1, S2, S3, B} <:
               SemImply
    Σ::A1
    A::A2
    S::A3
    F::A4
    μ::A5
    M::A6

    ram_matrices::V2
    has_meanstructure::B

    A_indices::I1
    S_indices::I2
    M_indices::I3

    F⨉I_A⁻¹::M1
    F⨉I_A⁻¹S::M2
    I_A::M3
    I_A⁻¹::M4

    ∇A::S1
    ∇S::S2
    ∇M::S3
end

using StructuralEquationModels

############################################################################################
### Constructors
############################################################################################

function RAM(;
    specification::SemSpecification,
    #vech = false,
    gradient = true,
    meanstructure = false,
    kwargs...,
)
    ram_matrices = convert(RAMMatrices, specification)

    # get dimensions of the model
    n_par = nparams(ram_matrices)
    n_obs = nobserved_vars(ram_matrices)
    n_var = nvars(ram_matrices)
    F = zeros(ram_matrices.size_F)
    F[CartesianIndex.(1:n_obs, ram_matrices.F_ind)] .= 1.0

    # get indices
    A_indices = copy(ram_matrices.A_ind)
    S_indices = copy(ram_matrices.S_ind)
    M_indices = !isnothing(ram_matrices.M_ind) ? copy(ram_matrices.M_ind) : nothing

    #preallocate arrays
    A_pre = zeros(n_var, n_var)
    S_pre = zeros(n_var, n_var)
    M_pre = !isnothing(M_indices) ? zeros(n_var) : nothing

    set_RAMConstants!(A_pre, S_pre, M_pre, ram_matrices.constants)

    A_pre = check_acyclic(A_pre, n_par, A_indices)

    # pre-allocate some matrices
    Σ = zeros(n_obs, n_obs)
    F⨉I_A⁻¹ = zeros(n_obs, n_var)
    F⨉I_A⁻¹S = zeros(n_obs, n_var)
    I_A = similar(A_pre)

    if gradient
        ∇A = matrix_gradient(A_indices, n_var^2)
        ∇S = matrix_gradient(S_indices, n_var^2)
    else
        ∇A = nothing
        ∇S = nothing
    end

    # μ
    if meanstructure
        has_meanstructure = Val(true)
        !isnothing(M_indices) || throw(
            ArgumentError(
                "You set `meanstructure = true`, but your model specification contains no mean parameters.",
            ),
        )
        ∇M = gradient ? matrix_gradient(M_indices, n_var) : nothing
        μ = zeros(n_obs)
    else
        has_meanstructure = Val(false)
        M_indices = nothing
        M_pre = nothing
        μ = nothing
        ∇M = nothing
    end

    return RAM(
        Σ,
        A_pre,
        S_pre,
        F,
        μ,
        M_pre,
        ram_matrices,
        has_meanstructure,
        A_indices,
        S_indices,
        M_indices,
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

# dispatch on meanstructure
objective!(imply::RAM, par, model::AbstractSemSingle) =
    objective!(imply, par, model, imply.has_meanstructure)
gradient!(imply::RAM, par, model::AbstractSemSingle) =
    gradient!(imply, par, model, imply.has_meanstructure)

# objective and gradient
function objective!(imply::RAM, params, model, has_meanstructure::Val{T}) where {T}
    fill_A_S_M!(
        imply.A,
        imply.S,
        imply.M,
        imply.A_indices,
        imply.S_indices,
        imply.M_indices,
        params,
    )

    @. imply.I_A = -imply.A
    @view(imply.I_A[diagind(imply.I_A)]) .+= 1

    copyto!(imply.F⨉I_A⁻¹, imply.F)
    rdiv!(imply.F⨉I_A⁻¹, factorize(imply.I_A))

    Σ_RAM!(imply.Σ, imply.F⨉I_A⁻¹, imply.S, imply.F⨉I_A⁻¹S)

    if T
        μ_RAM!(imply.μ, imply.F⨉I_A⁻¹, imply.M)
    end
end

function gradient!(
    imply::RAM,
    params,
    model::AbstractSemSingle,
    has_meanstructure::Val{T},
) where {T}
    fill_A_S_M!(
        imply.A,
        imply.S,
        imply.M,
        imply.A_indices,
        imply.S_indices,
        imply.M_indices,
        params,
    )

    @. imply.I_A = -imply.A
    @view(imply.I_A[diagind(imply.I_A)]) .+= 1

    imply.I_A⁻¹ = LinearAlgebra.inv!(factorize(imply.I_A))
    mul!(imply.F⨉I_A⁻¹, imply.F, imply.I_A⁻¹)

    Σ_RAM!(imply.Σ, imply.F⨉I_A⁻¹, imply.S, imply.F⨉I_A⁻¹S)

    if T
        μ_RAM!(imply.μ, imply.F⨉I_A⁻¹, imply.M)
    end
end

hessian!(imply::RAM, par, model::AbstractSemSingle, has_meanstructure) =
    gradient!(imply, par, model, has_meanstructure)
objective_gradient!(imply::RAM, par, model::AbstractSemSingle, has_meanstructure) =
    gradient!(imply, par, model, has_meanstructure)
objective_hessian!(imply::RAM, par, model::AbstractSemSingle, has_meanstructure) =
    gradient!(imply, par, model, has_meanstructure)
gradient_hessian!(imply::RAM, par, model::AbstractSemSingle, has_meanstructure) =
    gradient!(imply, par, model, has_meanstructure)
objective_gradient_hessian!(imply::RAM, par, model::AbstractSemSingle, has_meanstructure) =
    gradient!(imply, par, model, has_meanstructure)

############################################################################################
### Recommended methods
############################################################################################

function update_observed(imply::RAM, observed::SemObserved; kwargs...)
    if nobserved_vars(observed) == size(imply.Σ, 1)
        return imply
    else
        return RAM(; observed = observed, kwargs...)
    end
end

############################################################################################
### additional methods
############################################################################################

Σ(imply::RAM) = imply.Σ
μ(imply::RAM) = imply.μ

A(imply::RAM) = imply.A
S(imply::RAM) = imply.S
F(imply::RAM) = imply.F
M(imply::RAM) = imply.M

∇A(imply::RAM) = imply.∇A
∇S(imply::RAM) = imply.∇S
∇M(imply::RAM) = imply.∇M

A_indices(imply::RAM) = imply.A_indices
S_indices(imply::RAM) = imply.S_indices
M_indices(imply::RAM) = imply.M_indices

F⨉I_A⁻¹(imply::RAM) = imply.F⨉I_A⁻¹
F⨉I_A⁻¹S(imply::RAM) = imply.F⨉I_A⁻¹S
I_A(imply::RAM) = imply.I_A
I_A⁻¹(imply::RAM) = imply.I_A⁻¹ # only for gradient available!

has_meanstructure(imply::RAM) = imply.has_meanstructure

ram_matrices(imply::RAM) = imply.ram_matrices

############################################################################################
### additional functions
############################################################################################

function Σ_RAM!(Σ, F⨉I_A⁻¹, S, pre2)
    mul!(pre2, F⨉I_A⁻¹, S)
    mul!(Σ, pre2, F⨉I_A⁻¹')
end

function μ_RAM!(μ, F⨉I_A⁻¹, M)
    mul!(μ, F⨉I_A⁻¹, M)
end

function check_acyclic(A_pre, n_par, A_indices)
    # fill copy of A-matrix with random parameters
    A_rand = copy(A_pre)
    randpar = rand(n_par)

    fill_matrix!(A_rand, A_indices, randpar)

    # check if the model is acyclic
    acyclic = isone(det(I - A_rand))

    # check if A is lower or upper triangular
    if istril(A_rand)
        A_pre = LowerTriangular(A_pre)
    elseif istriu(A_rand)
        A_pre = UpperTriangular(A_pre)
    elseif acyclic
        @info "Your model is acyclic, specifying the A Matrix as either Upper or Lower Triangular can have great performance benefits.\n" maxlog =
            1
    end

    return A_pre
end
