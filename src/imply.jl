#imply is the new ram
abstract type Imply end

struct ImplyCommon{A <: AbstractArray} <: Imply
    implied::A
end

struct ImplySparse{A} <: Imply
    implied::A
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

    imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym .= ModelingToolkit.simplify.(imp_cov_sym)

    imp_fun_, imp_fun =
        ModelingToolkit.build_function(
            imp_cov_sym,
            parameters,
            expression=Val{false}
        )

    imp_cov = imp_fun_(start_val)
    return ImplySymbolic(imp_fun, imp_cov, start_val)
end

function(imply::ImplySymbolic)(parameters)
    imply.imp_fun(imply.imp_cov, parameters)
end

function (imply::ImplySparse)(par)
    imply.implied .= 0.0
end

function (imply::ImplyDense)(par)
    imply.implied .= 0.0
end

function (imply::ImplyCommon)(par)
    imply.implied .= 0.0
end
