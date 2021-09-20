using ModelingToolkit, SparseArrays, BenchmarkTools,
    LinearAlgebra

# Generate RAM matrices of factor models
function gen_model(nfact, nitem)
    nfact = Int64(nfact)
    nitem = Int64(nitem)

    ## Model definition
    nobs = Int64(nfact*nitem)
    nnod = nfact+nobs
    @ModelingToolkit.variables λ[1:nobs], σ[1:nobs]

    #F
    Ind = collect(1:nobs)
    Jnd = collect(1:nobs)
    V = fill(1,nobs)
    F = sparse(Ind, Jnd, V, nobs, nnod)

    #S
    Ind = collect(1:nnod)
    Jnd = collect(1:nnod)
    V = [σ; fill(1.0, nfact)]
    S = sparse(Ind, Jnd, V, nnod, nnod)


    #A
    Ind = collect(1:nobs)
    Jnd = vcat([fill(nobs+i, nitem) for i in 1:nfact]...)
    V = λ
    A = sparse(Ind, Jnd, V, nnod, nnod)
    return(A, S, F)
end

function neumann_series(mat::SparseMatrixCSC)
    inverse = I + mat
    next_term = mat^2

    while nnz(next_term) != 0
        inverse += next_term
        next_term *= mat
    end

    return inverse
end

function implied_cov(A, S, F)
    invia = neumann_series(A)

    imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym = ModelingToolkit.simplify.(imp_cov_sym)
    return imp_cov_sym
end

function symbolic_ULS(Sigma, S)
    diff = S - Sigma
    F = sum(x -> ^(x, 2), diff)
    #F = diff*permutedims(diff)
    #F = F[diagind(F)]
    F = simplify.(F)
    return diff
end

# small factor models
A, S, F = gen_model(1, 4)

Sigma = implied_cov(A, S, F)

nobs = size(Sigma, 1)

@ModelingToolkit.variables s[1:nobs, 1:nobs]

F = symbolic_ULS(Sigma, s)

