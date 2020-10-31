abstract type Imply end

struct ImplyCommon{A <: AbstractArray} <: Imply
    implied::A
end

struct ImplySparse{
    F1 <: Any,
    F2 <: Any,
    A <: AbstractArray,
    C <: SparseMatrixCSC,
    D <: SparseMatrixCSC,
    E <: SparseMatrixCSC,
    S <: Array{Float64}} <: Imply
    imp_fun_invia::F1
    imp_fun_S::F2
    F::C
    invia::D
    S::E
    imp_cov::A
    start_val::S
end

struct ImplyDense{A <: Array{Float64}} <: Imply
    implied::A
end

struct ImplySymbolic{
    F <: Any,
    A <: AbstractArray,
    S <: Array{Float64}} <: Imply
    imp_fun::F
    imp_cov::A
    start_val::S
end

mutable struct ImplySymbolicForward{
    F <: Any,
    S <: Array{Float64}} <: Imply
    imp_fun::F
    imp_cov
    start_val::S
end

mutable struct ImplySymbolicAlloc{
    F <: Any,
    S <: Array{Float64}} <: Imply
    imp_fun::F
    imp_cov
    start_val::S
end

function ImplySparse(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC
            }

    invia = I + A
    next_term = A^2

    while nnz(next_term) != 0
        invia += next_term
        next_term *= next_term
    end

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

    return ImplySparse( imp_fun_invia,
                        imp_fun_S,
                        F,
                        invia_pre,
                        S_pre,
                        imp_cov,
                        start_val)
end

function ImplySymbolic(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC
            }

    invia = I + A
    next_term = A^2

    while nnz(next_term) != 0
        invia += next_term
        next_term *= next_term
    end

    imp_cov_sym = F*invia*S*invia'*F'

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym .= ModelingToolkit.simplify.(imp_cov_sym)

    imp_fun =
        eval(ModelingToolkit.build_function(
            imp_cov_sym,
            parameters
        )[2])

    imp_cov = rand(size(F)[1], size(F)[1])
    #imp_cov = Base.invokelatest(imp_fun_, start_val)
    return ImplySymbolic(imp_fun, imp_cov, start_val)
end

function ImplySymbolicAlloc(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC
            }

    invia = I + A
    next_term = A^2

    while nnz(next_term) != 0
        invia += next_term
        next_term *= next_term
    end

    imp_cov_sym = F*invia*S*invia'*F'

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym .= ModelingToolkit.simplify.(imp_cov_sym)

    imp_fun =
        eval(ModelingToolkit.build_function(
            imp_cov_sym,
            parameters
        )[1])

    imp_cov = rand(size(F)[1], size(F)[1])
    #imp_cov = Base.invokelatest(imp_fun_, start_val)
    return ImplySymbolicAlloc(imp_fun, imp_cov, start_val)
end

function ImplySymbolicForward(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC
            }

    invia = I + A
    next_term = A^2

    while nnz(next_term) != 0
        invia += next_term
        next_term *= next_term
    end

    imp_cov_sym = F*invia*S*invia'*F'

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym .= ModelingToolkit.simplify.(imp_cov_sym)

    #imp_cov_sym = DiffEqBase.dualcache(imp_cov_sym)

    imp_fun =
        eval(ModelingToolkit.build_function(
            imp_cov_sym,
            parameters
        )[1])

    imp_cov = DiffEqBase.dualcache(zeros(size(F)[1], size(F)[1]))
    #imp_cov = Base.invokelatest(imp_fun_, start_val)
    return ImplySymbolicForward(imp_fun, imp_cov, start_val)
end

function(imply::ImplySymbolic)(parameters)
    imply.imp_fun(imply.imp_cov, parameters)
end

function(imply::ImplySymbolicAlloc)(parameters)
    imply.imp_cov = imply.imp_fun(parameters)
end

function(imply::ImplySymbolicForward)(parameters)
    imply.imp_cov = DiffEqBase.get_tmp(imply.imp_cov, parameters)
    imply.imp_fun(imply.imp_cov, parameters)
end

function (imply::ImplySparse)(parameters)
    imply.imp_fun_invia(imply.invia, parameters)
    imply.imp_fun_S(imply.S, parameters)
    #let (F, S, invia) = (imply.F, imply.S, imply.invia)
    imply.imp_cov .= imply.F*imply.invia*imply.S*imply.invia'*imply.F'
    #    imply.imp_cov .= Array(F*invia*S*invia'*F')
    #end
end

function (imply::ImplyDense)(par)
    imply.implied .= 0.0
end

function (imply::ImplyCommon)(par)
    imply.implied .= 0.0
end
