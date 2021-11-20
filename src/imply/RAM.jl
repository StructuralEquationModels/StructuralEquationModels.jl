############################################################################
### Types
############################################################################

struct RAMSymbolic{F1, F2, F3, A1, A2, A3, S1, S2, S3, V, F4, A4, F5, A5} <: SemImply
    Σ_function::F1
    ∇Σ_function::F2
    ∇²Σ_function::F3
    Σ::A1
    ∇Σ::A2
    ∇²Σ::A3
    Σ_symbolic::S1
    ∇Σ_symbolic::S2
    ∇²Σ_symbolic::S3
    start_val::V
    μ_function::F4
    μ::A4
    ∇μ_function::F5
    ∇μ::A5
end

############################################################################
### Constructors
############################################################################

function RAMSymbolic(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        par,
        start_val;
        M::Spa4 = nothing,
        vech = false,
        gradient = true,
        hessian = false
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC,
            Spa4 <: Union{Nothing, AbstractArray}
            }

    # Σ
    Σ_symbolic = get_Σ_symbolic_RAM(S, A, F; vech = vech)
    #print(Symbolics.build_function(Σ_symbolic)[2])
    Σ_function = eval(Symbolics.build_function(Σ_symbolic, par)[2])
    Σ = zeros(size(Σ_symbolic))

    # ∇Σ
    if gradient
        ∇Σ_symbolic = Symbolics.sparsejacobian(vec(Σ_symbolic), [par...])
        ∇Σ_function = eval(Symbolics.build_function(∇Σ_symbolic, par)[2])
        constr = findnz(∇Σ_symbolic)
        ∇Σ = sparse(constr[1], constr[2], fill(1.0, nnz(∇Σ_symbolic)), size(∇Σ_symbolic)...)
    else
        ∇Σ_symbolic = nothing
        ∇Σ_function = nothing
        ∇Σ = nothing
    end

    if hessian
        n_sig = length(Σ_symbolic)
        n_par = size(par, 1)
        ∇²Σ_symbolic_vec = [Symbolics.sparsehessian(σᵢ, [par...]) for σᵢ in vec(Σ_symbolic)]

        @variables J[1:n_sig]
        ∇²Σ_symbolic = zeros(Num, n_par, n_par)
        for i in 1:n_sig
            ∇²Σ_symbolic += J[i]*∇²Σ_symbolic_vec[i]
        end
    
        ∇²Σ_function = eval(Symbolics.build_function(∇²Σ_symbolic, J, par)[2])
        ∇²Σ = zeros(n_par, n_par)
    else
        ∇²Σ_symbolic = nothing
        ∇²Σ_function = nothing
        ∇²Σ = nothing
    end

    # μ
    if !isnothing(M)
        μ_symbolic = get_μ_symbolic_RAM(M, A, F)
        μ_function = eval(Symbolics.build_function(μ_symbolic, par)[2])
        μ = zeros(size(μ_symbolic))
        if gradient
            ∇μ_symbolic = Symbolics.jacobian(μ_symbolic, par)
            ∇μ_function = eval(Symbolics.build_function(∇μ_symbolic, par)[2])
            ∇μ = zeros(size(F, 1), size(par, 1))
        else
            ∇μ_function = nothing
            ∇μ = nothing
        end
    else
        μ_function = nothing
        μ = nothing
        ∇μ_function = nothing
        ∇μ = nothing
    end

    return RAMSymbolic(
        Σ_function,
        ∇Σ_function,
        ∇²Σ_function,
        Σ,
        ∇Σ,
        ∇²Σ,
        Σ_symbolic,
        ∇Σ_symbolic,
        ∇²Σ_symbolic,
        copy(start_val),
        μ_function,
        μ,
        ∇μ_function,
        ∇μ
    )
end

############################################################################
### functors
############################################################################

function (imply::RAMSymbolic)(par, F, G, H, model)
    imply.Σ_function(imply.Σ, par)
    if !isnothing(G) || !isnothing(H)
        imply.∇Σ_function(imply.∇Σ, par)
    end
    if !isnothing(imply.μ)
        imply.μ_function(imply.μ, par)
        if !isnothing(G) || !isnothing(H)
            imply.∇μ_function(imply.∇μ, par)
        end
    end
end