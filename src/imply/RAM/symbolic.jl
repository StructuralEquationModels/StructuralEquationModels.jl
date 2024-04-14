############################################################################################
### Types
############################################################################################
@doc raw"""
Subtype of `SemImply` that implements the RAM notation with symbolic precomputation.

# Constructor

    RAMSymbolic(;specification,
        vech = false,
        gradient = true,
        hessian = false,
        approximate_hessian = false,
        meanstructure = false,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object
- `meanstructure::Bool`: does the model have a meanstructure?
- `gradient::Bool`: is gradient-based optimization used
- `hessian::Bool`: is hessian-based optimization used
- `approximate_hessian::Bool`: for hessian based optimization: should the hessian be approximated
- `vech::Bool`: should the half-vectorization of Σ be computed (instead of the full matrix)
    (automatically set to true if any of the loss functions is SemWLS)

# Extended help

## Implementation
Subtype of `SemImply`.

## Interfaces
- `identifier(::RAMSymbolic) `-> Dict containing the parameter labels and their position
- `nparams(::RAMSymbolic)` -> number of parameters

- `Σ(::RAMSymbolic)` -> model implied covariance matrix
- `μ(::RAMSymbolic)` -> model implied mean vector

Jacobians (only available in gradient! calls)
- `∇Σ(::RAMSymbolic)` -> ``∂vec(Σ)/∂θᵀ``
- `∇μ(::RAMSymbolic)` -> ``∂μ/∂θᵀ``

- `∇Σ_function(::RAMSymbolic)` -> function to overwrite `∇Σ` in place,
    i.e. `∇Σ_function(∇Σ, θ)`. Normally, you do not want to use this but simply
    query `∇Σ(::RAMSymbolic)`.

Hessians
The computation of hessians is more involved, and uses the "chain rule for
hessian matrices".
Therefore, we desribe it at length in the mathematical appendix of the online documentation,
and the relevant interfaces are omitted here.

Additional interfaces
- `has_meanstructure(::RAMSymbolic)` -> `Val{Bool}` does the model have a meanstructure?

## RAM notation
The model implied covariance matrix is computed as
```math
    \Sigma = F(I-A)^{-1}S(I-A)^{-T}F^T
```
and for models with a meanstructure, the model implied means are computed as
```math
    \mu = F(I-A)^{-1}M
```
"""
struct RAMSymbolic{MS, F1, F2, F3, A1, A2, A3, S1, S2, S3, V2, F4, A4, F5, A5, D1} <: SemImplySymbolic{MS,ExactHessian}
    Σ_function::F1
    ∇Σ_function::F2
    ∇²Σ_function::F3
    Σ::A1
    ∇Σ::A2
    ∇²Σ::A3
    Σ_symbolic::S1
    ∇Σ_symbolic::S2
    ∇²Σ_symbolic::S3
    ram_matrices::V2
    μ_function::F4
    μ::A4
    ∇μ_function::F5
    ∇μ::A5
    identifier::D1
end

############################################################################################
### Constructors
############################################################################################

RAMSymbolic{MS}(args...) where MS <: MeanStructure = RAMSymbolic{MS, map(typeof, args)...}(args...)

function RAMSymbolic(;
        specification::SemSpecification,
        loss_types = nothing,
        vech = false,
        gradient = true,
        hessian = false,
        meanstructure = false,
        approximate_hessian = false,
        kwargs...)

    ram_matrices = convert(RAMMatrices, specification)

    n_par = nparams(ram_matrices)
    n_obs = nobserved_vars(ram_matrices)
    n_var = nvars(ram_matrices)

    par = (Symbolics.@variables θ[1:n_par])[1]

    A = zeros(Num, n_var, n_var)
    S = zeros(Num, n_var, n_var)
    !isnothing(ram_matrices.M_ind) ? M = zeros(Num, n_var) : M = nothing
    F = zeros(ram_matrices.size_F); F[CartesianIndex.(1:n_obs, ram_matrices.F_ind)] .= 1.0

    set_RAMConstants!(A, S, M, ram_matrices.constants)
    fill_A_S_M!(A, S, M, ram_matrices.A_ind, ram_matrices.S_ind, ram_matrices.M_ind, par)

    A, S, F = sparse(A), sparse(S), sparse(F)

    if !isnothing(loss_types)
        any(loss_types .<: SemWLS) ? vech = true : nothing
    end

    # Σ
    Σ_symbolic = get_Σ_symbolic_RAM(S, A, F; vech = vech)
    #print(Symbolics.build_function(Σ_symbolic)[2])
    Σ_function = Symbolics.build_function(Σ_symbolic, par, expression=Val{false})[2]
    Σ = zeros(size(Σ_symbolic))
    precompile(Σ_function, (typeof(Σ), Vector{Float64}))

    # ∇Σ
    if gradient
        ∇Σ_symbolic = Symbolics.sparsejacobian(vec(Σ_symbolic), [par...])
        ∇Σ_function = Symbolics.build_function(∇Σ_symbolic, par, expression=Val{false})[2]
        constr = findnz(∇Σ_symbolic)
        ∇Σ = sparse(constr[1], constr[2], fill(1.0, nnz(∇Σ_symbolic)), size(∇Σ_symbolic)...)
        precompile(∇Σ_function, (typeof(∇Σ), Vector{Float64}))
    else
        ∇Σ_symbolic = nothing
        ∇Σ_function = nothing
        ∇Σ = nothing
    end

    if hessian && !approximate_hessian
        n_sig = length(Σ_symbolic)
        ∇²Σ_symbolic_vec = [Symbolics.sparsehessian(σᵢ, [par...]) for σᵢ in vec(Σ_symbolic)]

        @variables J[1:n_sig]
        ∇²Σ_symbolic = zeros(Num, n_par, n_par)
        for i in 1:n_sig
            ∇²Σ_symbolic += J[i]*∇²Σ_symbolic_vec[i]
        end
    
        ∇²Σ_function = Symbolics.build_function(∇²Σ_symbolic, J, par, expression=Val{false})[2]
        ∇²Σ = zeros(n_par, n_par)
    else
        ∇²Σ_symbolic = nothing
        ∇²Σ_function = nothing
        ∇²Σ = nothing
    end

    # μ
    if meanstructure
        MS = HasMeanStructure
        μ_symbolic = get_μ_symbolic_RAM(M, A, F)
        μ_function = Symbolics.build_function(μ_symbolic, par, expression=Val{false})[2]
        μ = zeros(size(μ_symbolic))
        if gradient
            ∇μ_symbolic = Symbolics.jacobian(μ_symbolic, [par...])
            ∇μ_function = Symbolics.build_function(∇μ_symbolic, par, expression=Val{false})[2]
            ∇μ = zeros(size(F, 1), size(par, 1))
        else
            ∇μ_function = nothing
            ∇μ = nothing
        end
    else
        MS = NoMeanStructure
        μ_function = nothing
        μ = nothing
        ∇μ_function = nothing
        ∇μ = nothing
    end

    return RAMSymbolic{MS}(
        Σ_function,
        ∇Σ_function,
        ∇²Σ_function,
        Σ,
        ∇Σ,
        ∇²Σ,
        Σ_symbolic,
        ∇Σ_symbolic,
        ∇²Σ_symbolic,
        ram_matrices,
        μ_function,
        μ,
        ∇μ_function,
        ∇μ,
        identifier(ram_matrices),
    )
end

############################################################################################
### objective, gradient, hessian
############################################################################################

function update!(targets::EvaluationTargets, imply::RAMSymbolic, model::AbstractSemSingle, par)
    imply.Σ_function(imply.Σ, par)
    if MeanStructure(imply) === HasMeanStructure
        imply.μ_function(imply.μ, par)
    end

    if is_gradient_required(targets) || is_hessian_required(targets)
        imply.∇Σ_function(imply.∇Σ, par)
        if MeanStructure(imply) === HasMeanStructure
            imply.∇μ_function(imply.∇μ, par)
        end
    end
end

############################################################################################
### Recommended methods
############################################################################################

identifier(imply::RAMSymbolic) = imply.identifier
nparams(imply::RAMSymbolic) = nparams(imply.ram_matrices)

function update_observed(imply::RAMSymbolic, observed::SemObserved; kwargs...)
    if Int(n_man(observed)) == size(imply.Σ, 1)
        return imply
    else
        return RAMSymbolic(;observed = observed, kwargs...)
    end
end

############################################################################################
### additional functions
############################################################################################

function get_Σ_symbolic_RAM(S, A, F; vech = false)
    invia = neumann_series(A)
    Σ_symbolic = F*invia*S*permutedims(invia)*permutedims(F)
    Σ_symbolic = Array(Σ_symbolic)
    # Σ_symbolic = Symbolics.simplify.(Σ_symbolic)
    Threads.@threads for i in eachindex(Σ_symbolic)
        Σ_symbolic[i] = Symbolics.simplify(Σ_symbolic[i])
    end
    if vech Σ_symbolic = Σ_symbolic[tril(trues(size(F, 1), size(F, 1)))] end
    return Σ_symbolic
end

function get_μ_symbolic_RAM(M, A, F)
    invia = neumann_series(A)
    μ_symbolic = F*invia*M
    μ_symbolic = Array(μ_symbolic)
    Threads.@threads for i in eachindex(μ_symbolic)
        μ_symbolic[i] = Symbolics.simplify(μ_symbolic[i])
    end
    return μ_symbolic
end