struct ImplySparse{
    F1 <: Any,
    F2 <: Any,
    A <: AbstractArray,
    C <: SparseMatrixCSC,
    D <: SparseMatrixCSC,
    E <: SparseMatrixCSC,
    S <: Array{Float64},
    A2 <: Union{Nothing, AbstractArray}} <: Imply
    imp_fun_invia::F1
    imp_fun_S::F2
    F::C
    invia::D
    S::E
    imp_cov::A
    start_val::S
    imp_mean::A2
end

function ImplySparse(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val;
        M = nothing
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC
            }

    invia = neumann_series(A)

    #imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
    #imp_cov_sym = Array(imp_cov_sym)
    invia .= ModelingToolkit.simplify.(invia)

    imp_fun_invia_, imp_fun_invia =
        eval.(ModelingToolkit.build_function(
            invia,
            parameters
            ))

    imp_fun_S_, imp_fun_S =
        eval.(ModelingToolkit.build_function(
            S,
            parameters
            ))

    invia_pre = Base.invokelatest(imp_fun_invia_, start_val)
    S_pre = Base.invokelatest(imp_fun_S_, start_val)
    imp_cov = rand(size(F)[1], size(F)[1])

    if isnothing(M)
        imp_mean = nothing
    else
    end

    return ImplySparse( imp_fun_invia,
                        imp_fun_S,
                        copy(F),
                        invia_pre,
                        S_pre,
                        imp_cov,
                        copy(start_val),
                        imp_mean)
end


function (imply::ImplySparse)(parameters, model)
    imply.imp_fun_invia(imply.invia, parameters)
    imply.imp_fun_S(imply.S, parameters)
    imp_cov = Matrix(imply.invia*imply.S*transpose(imply.invia))
    nvar = size(imply.F, 1)
    imply.imp_cov .= imp_cov[1:nvar, 1:nvar]
end



#################
struct ImplyCommon{
    C <: Real,
    D <: AbstractArray,
    E <: AbstractArray,
    S <: Array{Float64},
    A2 <: Union{Nothing, AbstractArray},
    A <: AbstractArray,
    F1 <: Any,
    F2 <: Any} <: Imply
    n_var::C
    imp_fun_A::F1
    imp_fun_S::F2
    A::D
    S::E
    imp_cov::A
    start_val::S
    imp_mean::A2
end

function ImplyCommon(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val;
        M = nothing
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC
            }

    A = Matrix(I-A)
    S = Matrix(S)

    imp_fun_A_, imp_fun_A =
        eval.(ModelingToolkit.build_function(
            A,
            parameters
            ))

    imp_fun_S_, imp_fun_S =
        eval.(ModelingToolkit.build_function(
            S,
            parameters
            ))

    A_pre = Base.invokelatest(imp_fun_A_, start_val)
    S_pre = Base.invokelatest(imp_fun_S_, start_val)
    imp_cov = rand(size(F)[1], size(F)[1])

    n_var = float(size(F, 1))

    if isnothing(M)
        imp_mean = nothing
    else
    end

    return ImplyCommon( n_var,
                        imp_fun_A,
                        imp_fun_S,
                        A_pre,
                        S_pre,
                        imp_cov,
                        copy(start_val),
                        imp_mean)
end

function (imply::ImplyCommon)(parameters, model)
    imply.imp_fun_A(imply.A, parameters)
    imply.imp_fun_S(imply.S, parameters)
    invia = inv(imply.A)
    imp_cov = invia*imply.S*transpose(invia)
    #imply.imp_cov .= imp_cov[1:imply.n_var, 1:imply.n_var]
    copyto!(imply.imp_cov, CartesianIndices(imply.imp_cov), imp_cov, CartesianIndices((imply.imp_cov)))
end