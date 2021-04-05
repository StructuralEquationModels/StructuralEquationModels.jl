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

    return ImplySparse( imp_fun_invia,
                        imp_fun_S,
                        copy(F),
                        invia_pre,
                        S_pre,
                        imp_cov,
                        copy(start_val))
end


function (imply::ImplySparse)(parameters, model)
    imply.imp_fun_invia(imply.invia, parameters)
    imply.imp_fun_S(imply.S, parameters)
    #let (F, S, invia) = (imply.F, imply.S, imply.invia)
    imply.imp_cov .= imply.F*imply.invia*imply.S*imply.invia'*imply.F'
    #    imply.imp_cov .= Array(F*invia*S*invia'*F')
    #end
end
