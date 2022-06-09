############################################################################################
### Types
############################################################################################
mutable struct RAMSymbolicZ{F1, F2, A1, A2, S1, S2, V, V2, F4, A4, F5, A5, D1, B} <: SemImplySymbolic
    Σ_function::F1
    ∇Σ_function::F2
    Σ::A1
    ∇Σ::A2
    Σ_symbolic::S1
    ∇Σ_symbolic::S2
    n_par::V
    ram_matrices::V2
    μ_function::F4
    μ::A4
    ∇μ_function::F5
    ∇μ::A5
    identifier::D1
    has_meanstructure::B
end

############################################################################################
### Constructors
############################################################################################

function RAMSymbolicZ(;
        specification,
        loss_types = nothing,
        vech = false,
        gradient = true,
        meanstructure = false,
        kwargs...)

    ram_matrices = RAMMatrices(specification)
    identifier = StructuralEquationModels.identifier(ram_matrices)

    n_par = length(ram_matrices.parameters)
    n_var, n_nod = ram_matrices.size_F

    par = (Symbolics.@variables θ[1:n_par])[1]

    A = zeros(Num, n_nod, n_nod)
    S = zeros(Num, n_nod, n_nod)
    !isnothing(ram_matrices.M_ind) ? M = zeros(Num, n_nod) : M = nothing
    F = zeros(ram_matrices.size_F); F[CartesianIndex.(1:n_var, ram_matrices.F_ind)] .= 1.0

    set_RAMConstants!(A, S, M, ram_matrices.constants)
    fill_A_S_M(A, S, M, ram_matrices.A_ind, ram_matrices.S_ind, ram_matrices.M_ind, par)

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

    # μ
    if meanstructure
        has_meanstructure = Val(true)
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
        has_meanstructure = Val(false)
        μ_function = nothing
        μ = nothing
        ∇μ_function = nothing
        ∇μ = nothing
    end

    return RAMSymbolicZ(
        Σ_function,
        ∇Σ_function,
        Σ,
        ∇Σ,
        Σ_symbolic,
        ∇Σ_symbolic,
        n_par,
        ram_matrices,
        μ_function,
        μ,
        ∇μ_function,
        ∇μ,
        identifier,
        has_meanstructure
    )
end

############################################################################################
### objective, gradient, hessian
############################################################################################

# dispatch on meanstructure
objective!(imply::RAMSymbolicZ, par, model) = 
    objective!(imply, par, model, imply.has_meanstructure)

# objective
function objective!(imply::RAMSymbolicZ, par, model, has_meanstructure::Val{T}) where T
    Σ = update_Σ_wrap(imply.Σ, par, imply.Σ_function, imply.∇Σ_function, imply.∇Σ)
    imply.Σ = Σ
    T && imply.μ_function(imply.μ, par)
end

############################################################################################
### functions
############################################################################################

function update_Σ_wrap(Σ_pre, par, Σ_function, ∇Σ_function, ∇Σ)
    Σ = similar(Σ_pre)
    Σ_function(Σ, par)
    return Σ
end

function ChainRulesCore.rrule(::typeof(update_Σ_wrap), Σ_pre, par, Σ_function, ∇Σ_function, ∇Σ)
    Σ = update_Σ_wrap(Σ_pre, par, Σ_function, ∇Σ_function, ∇Σ)
    ∇Σ_function(∇Σ, par)
    function update_Σ_wrap_pullback(ȳ)
        f̄ = NoTangent()
        Σ̄_pre = NoTangent()
        Σ̄_function = NoTangent()
        p̄ar = @thunk((vec(ȳ)'*∇Σ)')
        ∇Σ̄_function = NoTangent()
        ∇Σ̄ = NoTangent()
        return f̄, Σ̄_pre, p̄ar, Σ̄_function, ∇Σ̄_function, ∇Σ̄
    end
    return Σ, update_Σ_wrap_pullback
end

############################################################################################
### additional methods
############################################################################################

Σ(imply::RAMSymbolicZ) = imply.Σ
∇Σ(imply::RAMSymbolicZ) = imply.∇Σ

μ(imply::RAMSymbolicZ) = imply.μ
∇μ(imply::RAMSymbolicZ) = imply.∇μ

Σ_function(imply::RAMSymbolicZ) = imply.Σ_function
∇Σ_function(imply::RAMSymbolicZ) = imply.∇Σ_function

has_meanstructure(imply::RAMSymbolicZ) = imply.has_meanstructure

identifier(imply::RAMSymbolicZ) = imply.identifier
n_par(imply::RAMSymbolicZ) = imply.n_par