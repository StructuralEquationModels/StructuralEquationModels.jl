############################################################################
### Types
############################################################################

struct ImplySymbolicSWLS{
    F <: Any,
    A <: AbstractArray,
    S <: Array{Float64}} <: Imply
    imp_fun::F
    G::A
    start_val::S
end

############################################################################
### Constructors
############################################################################

function ImplySymbolicSWLS(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC,
            Spa4 <: Union{Nothing, AbstractArray}
            }

    nobs, ntot = size(F)
    nlat = ntot - nobs

    # Selection matrices
    ind = findall(tril(trues(nobs, nobs)))
    Es = []
    for i in ind
        E = zeros(Bool, nobs, nobs)
        E[i] = true
        E = vec(E)
        push!(Es, E)
    end
    L = hcat(Es...)
    K = L*inv(transpose(L)*L)
    K = sparse(K)

    ind = findall(!iszero, S)
    ind = filter(x -> (x[1] >= x[2]), ind)
    Es = []
    for i in ind
        E = zeros(Bool, nobs+nlat, nobs+nlat)
        E[i] = true
        E = vec(E)
        push!(Es, E)
    end
    L_Ω = hcat(Es...)
    L_Ω = sparse(L_Ω)

    #Model-implied covmat
    A = neumann_series(A)

    G = transpose(K)*(kron(F, F)*kron(A, A))*L_Ω
    G = Array(G)
    G = ModelingToolkit.simplify.(G)

    imp_fun =
        eval(ModelingToolkit.build_function(
            G,
            parameters
        )[2])

    G = zeros(size(G))

    return ImplySymbolicSWLS(
        imp_fun,
        G,
        copy(start_val)
    )
end

############################################################################
### functors
############################################################################

function (imply::ImplySymbolicSWLS)(parameters, model)
    imply.imp_fun(imply.G, parameters)
end

############################################################################
### additional functions
############################################################################