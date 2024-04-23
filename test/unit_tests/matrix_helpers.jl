using StructuralEquationModels, Test, Random, SparseArrays, LinearAlgebra
using StructuralEquationModels:
    CommutationMatrix, transpose_linear_indices, duplication_matrix, elimination_matrix

Random.seed!(73721)

n = 4
m = 5

@testset "Commutation matrix" begin
    # transpose linear indices
    A = rand(n, m)
    @test reshape(A[transpose_linear_indices(n, m)], m, n) == A'
    # commutation matrix multiplication
    K = CommutationMatrix(n)
    B = rand(n, n)
    @test K * vec(B) == vec(B')
    C = sprand(n, n, 0.5)
    @test K * vec(C) == vec(C')
    # lmul!
    D = sprand(n^2, n^2, 0.1)
    E = copy(D)
    lmul!(K, D)
    @test D == K * E
end

@testset "Duplication / elimination matrix" begin
    A = rand(m, m)
    A = A * A'
    # dupication
    D = duplication_matrix(m)
    @test D * A[tril(trues(size(A)))] == vec(A)
    # elimination
    D = elimination_matrix(m)
    @test D * vec(A) == A[tril(trues(size(A)))]
end
