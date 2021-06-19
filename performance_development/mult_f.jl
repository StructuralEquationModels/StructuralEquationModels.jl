using SparseArrays, LinearAlgebra, BenchmarkTools, MKL

nobs = 200
nlat = 30
A = rand(nobs+nlat, nobs+nlat)
F = zeros(nobs, nobs+nlat)
for i = 1:nobs F[i, i] = 1.0 end

@benchmark F*A

pre = similar(F)

@benchmark copyto!(pre, CartesianIndices(pre), A, CartesianIndices(pre))
