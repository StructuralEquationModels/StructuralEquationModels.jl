############################################################################
### Types
############################################################################

struct RAMSymbolic{F1, F2, A1, A2, S1, S2, V, F3, A3} <: Imply
    Σ_function::F1
    ∇Σ_function::F2
    Σ::A1
    ∇Σ::A2
    Σ_symbolic::S1
    ∇Σ_symbolic::S2
    start_val::V
    μ_function::F3
    μ::A3
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

    par = [par[i] for i in size(par, 1)]

    # Σ
    Σ_symbolic = get_Σ_symbolic_RAM(S, A, F; vech = vech)
    Σ_function = eval(ModelingToolkit.build_function(Σ_symbolic, par)[2])
    Σ = zeros(size(F)[1], size(F)[1])

    # ∇Σ
    if gradient
        ∇Σ_symbolic = ModelingToolkit.jacobian(vec(imp_cov_sym), par)
        ∇Σ_function = eval(ModelingToolkit.build_function(∇Σ_symbolic, parameters)[2])
        ∇Σ = zeros(size(F, 1)^2, size(parameters, 1))
    else
        ∇Σ_symbolic = nothing
        ∇Σ_function = nothing
        ∇Σ = nothing
    end

    # μ
    if !isnothing(M)
        stop("means are not implemented yet")
#=         imp_mean_sym = F*invia*M
        imp_mean_sym = Array(imp_mean_sym)
        imp_mean_sym = ModelingToolkit.simplify.(imp_mean_sym)

        imp_fun_mean = eval(ModelingToolkit.build_function(imp_mean_sym, parameters)[2])

        imp_mean = zeros(size(F)[1]) =#
    else
        μ_function = nothing
        μ = nothing
    end

    return RAMSymbolic(
        Σ_function,
        ∇Σ_function,
        Σ,
        ∇Σ,
        Σ_symbolic,
        ∇Σ_symbolic,
        copy(start_val),
        μ_function,
        μ
    )
end

############################################################################
### functors
############################################################################

function (imply::RAMSymbolic)(par, F, G, H, model)
    imply.Σ_function(imply.Σ, par)
    if !isnothing(imply.imp_mean)
        imply.imp_fun_mean(imply.imp_mean, parameters)
    end
    if !isnothing(G)
        imply.∇Σ_function(imply.∇Σ, par)
    end
    # if !isnothing(H)
    #     imply.∇²Σ_function(imply.∇²Σ, par)
    # end
end