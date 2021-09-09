using InvertedIndices, BenchmarkTools, LinearAlgebra, ProfileView,
    GraphRecipes

test = rand(10, 1)
@benchmark LinearAlgebra.BLAS.gemm('N', 'T', 1.0, test, test)
@benchmark test*test'

A = rand(10,10)
B = rand(10,10)
C = A\B

S_inv = rand(20,20)
S_inv = S_inv*S_inv'
S = Symmetric(S)
res = inv(S_inv[Not(5), Not(5)])
S_inv = Symmetric(inv(S_inv))

h = S_inv[5, 5]
a = S_inv[Not(5), 5] / sqrt(h)
out = S_inv[Not(5), Not(5)] - a*a'

res ≈ out

## rank 1 update
function sym_inv_update(M_inv, ind_not::Int64, ind)
    h = M_inv[ind_not, ind_not]
    a = M_inv[ind, [ind_not]] / sqrt(h)
    out = M_inv[ind, ind] - LinearAlgebra.BLAS.gemm('N', 'T', 1.0, a, a)
    return out
end

# rank-n-update
function sym_inv_update(M_inv, ind_not, ind)
    A = M_inv[ind_not, ind]
    H = cholesky(M_inv[ind_not, ind_not])
    D = H \ A
    out = M_inv[ind, ind] - LinearAlgebra.BLAS.gemm('T', 'N', 1.0, A, D)
    return out
end

S = rand(50,50)
S = S*S'
S = Symmetric(S)
S_inv = Symmetric(inv(S))

ind = filter(x ->!(x ∈ [5]), 1:size(S, 1))

sym_inv_update(S_inv, 5, ind) ≈ inv(S[Not(5), Not(5)])
sym_inv_update(S_inv, [5], ind) ≈ inv(S[Not(5), Not(5)])

@benchmark inv($S[ind, ind])
@benchmark sym_inv_update($S_inv, 5, ind)
@benchmark sym_inv_update($S_inv, [5], ind)

ind = filter(x ->!(x ∈ [2, 3, 5, 7]), 1:size(S, 1))

sym_inv_update(S_inv, [2,3, 5, 7], ind) ≈ inv(S[ind, ind])
inv(cholesky(S)) ≈ cholesky(S) \ I

@benchmark cholesky($S[ind, ind]) \ I
@benchmark sym_inv_update($S_inv, [2, 3, 5, 7], $ind)

function testf(S_inv, ind_not, ind)
    for i = 1:10000
        sym_inv_update(S_inv, ind_not, ind)
    end
end

ProfileView.@profview testf(S_inv, [5, 7], ind)


a = rand(40, 40)
b = rand(40, 40)

@benchmark a-b


#### random graph generation
using Random, LinearAlgebra, SparseArrays, GraphRecipes, Plots

Random.seed!(78435472956284237434)

nvar = 500
#p_con = 0.1
e_n = 4

p_con = (e_n/(nvar-1))

S = zeros(nvar, nvar)
S[diagind(S)] .= 1.0


# DAG
G = rand(nvar, nvar)
G = G .< p_con
G = LowerTriangular(G)
G[diagind(G)] .= false
G = sparse(G)

inv(I-Matrix(G))
C = C .!= 0
(C .& C') == I
@benchmark C = inv(I-Matrix(G))

# Cyclic
G = rand(nvar, nvar)
G = G .< p_con
G = sparse(G)

C = inv(I-Matrix(G))
C = C .!= 0
(C .& C') == I
@benchmark C = inv(I-Matrix(G))


graphplot(G)
print(eigvals(Matrix(G)))

## Fill with effects
ind = findall(G)
effects = rand(size(ind, 1))

A = zeros(nvar, nvar)
A[ind] .= effects
A[ind] .= 0.3

inv(I-A)*S*inv(I-A)'


### generate alternative Graphs 

A = [

    ]