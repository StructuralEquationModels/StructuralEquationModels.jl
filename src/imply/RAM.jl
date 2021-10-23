############################################################################
### Types
############################################################################

struct RAMSymbolic{
    F <: Any,
    F3 <: Any,
    A <: AbstractArray,
    A3 <: AbstractArray,
    S <: Array{Float64},
    F2 <: Any,
    A2 <: Union{Nothing, AbstractArray}} <: Imply
    imp_fun::F
    gradient_fun::F3
    imp_cov::A
    ∇Σ::A3
    start_val::S
    imp_fun_mean::F2
    imp_mean::A2
end

############################################################################
### Constructors
############################################################################

function RAMSymbolic(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val;
        M::Spa4 = nothing,
        vech = false
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC,
            Spa4 <: Union{Nothing, AbstractArray}
            }

    #Model-implied covmat

    invia = neumann_series(A)

    imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym = ModelingToolkit.simplify.(imp_cov_sym)

    imp_fun =
        eval(ModelingToolkit.build_function(
            imp_cov_sym,
            parameters
        )[2])

    imp_cov = zeros(size(F)[1], size(F)[1])

    ∇Σ_sym = ModelingToolkit.jacobian(vec(imp_cov_sym), parameters)
    gradient_fun =
        eval(ModelingToolkit.build_function(
                ∇Σ_sym,
                parameters
            )[2])
    ∇Σ = zeros(size(F, 1)^2, size(parameters, 1))
    #imp_cov = Base.invokelatest(imp_fun_, start_val)
    #Model implied mean
    if !isnothing(M)
        imp_mean_sym = F*invia*M
        imp_mean_sym = Array(imp_mean_sym)
        imp_mean_sym = ModelingToolkit.simplify.(imp_mean_sym)

        imp_fun_mean =
            eval(ModelingToolkit.build_function(
                imp_mean_sym,
                parameters
            )[2])

        imp_mean = zeros(size(F)[1])
    else
        imp_fun_mean = nothing
        imp_mean = nothing
    end

    return RAMSymbolic(
        imp_fun,
        gradient_fun,
        imp_cov,
        ∇Σ,
        copy(start_val),
        imp_fun_mean,
        imp_mean
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
    if !isnothing(H)
        imply.∇²Σ_function(imply.∇²Σ, par)
    end
end