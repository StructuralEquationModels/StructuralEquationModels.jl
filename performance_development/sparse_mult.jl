using SparseArrays, LinearAlgebra, BenchmarkTools, MKL

A = rand(200,200)
B = rand(200,200)

C = zeros(200,200)
C[3, 6] = 1
C[30, 60] = 1
C[31, 16] = 1
C[32, 56] = 1
C[43, 46] = 1
C[109, 89] = 1

ind = findall(x -> isone(x), C)

### BLAS
function genmul(A, C, B)
    D = A*C*B
    return D
end

@benchmark genmul(A, C, B)

C = sparse(C)

### SPARSE
@benchmark genmul(A, C, B)


### LOOP, column layout, without boradcasting, with @inbounds
B_2 = permutedims(B)

function outer_mul_4(A, B, ind)
    C = zeros(size(A))
    @inbounds for i in 1:length(ind)
        C += A[:, ind[i][1]]*B[:, ind[i][2]]'
    end
    return C
end

@benchmark outer_mul_4(A,B_2, ind)

### call BLAS directly
function outer_mul_5(A, B, ind)
    C = zeros(size(A))
    for i in 1:length(ind)
        BLAS.ger!(1.0, A[:, ind[i][1]], B[:, ind[i][2]], C)
    end
    return C
end

@benchmark outer_mul_5(A,B_2, ind)

### call BLAS directly, witout column mayor --> WINNER!!!
function outer_mul_7(A, B, ind)
    C = zeros(size(A))
    for i in 1:length(ind)
        BLAS.ger!(1.0, A[:, ind[i][1]], B[ind[i][2], :], C)
    end
    return C
end

@benchmark outer_mul_7(A,B, ind)

### with mul!
C = Matrix(C)

function outer_mul_6(A, B, C)
    D = similar(A)
    E = similar(C)
    mul!(D, A, C)
    mul!(E, D, B)
    return E
end

@benchmark outer_mul_6(A,B, C)
