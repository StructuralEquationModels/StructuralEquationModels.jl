using sem, Test, ModelingToolkit, LinearAlgebra, SparseArrays

## Test Matrices

S =[1 0 0 0 0 0 0 0 0 0 0 0.0
    0 1 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0
    0 0 0 0 0 0 0 0 0 1 0.5 0.5
    0 0 0 0 0 0 0 0 0 0.5 1 0.5
    0 0 0 0 0 0 0 0 0 0.5 0.5 1]

F =[1.0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0]

A =[0  0  0  0  0  0  0  0  0  1     0     0.0
    0  0  0  0  0  0  0  0  0  0.5   0     0
    0  0  0  0  0  0  0  0  0  0.5   0     0
    0  0  0  0  0  0  0  0  0  0     1     0
    0  0  0  0  0  0  0  0  0  0     0.5   0
    0  0  0  0  0  0  0  0  0  0     0.5   0
    0  0  0  0  0  0  0  0  0  0     0     1
    0  0  0  0  0  0  0  0  0  0     0     0.5
    0  0  0  0  0  0  0  0  0  0     0     0.5
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0]

imp_cov_start = F*inv(I-A)*S*transpose(inv(I-A))*transpose(F)

A[2,10] = 1; A[3,10] = 1; A[5,11] = 1; A[6,11] = 1; A[8,12] = 1; A[9,12] = 1

S[10, 11] = 1; S[10, 12] = 1; S[11, 10] = 1;
S[11, 12] = 1; S[12, 10] = 1; S[12, 11] = 1

imp_cov_new = F*inv(I-A)*S*transpose(inv(I-A))*transpose(F)

@variables x[1:30]

#S
Ind = [collect(1:12); 10; 10; 11; 11; 12; 12]
Jnd = [collect(1:12); 11; 12; 10; 12; 10; 11]
V = [[x[i] for i = 1:14]; x[13]; x[15]; x[14]; x[15]]
S = sparse(Ind, Jnd, V)

#F
Ind = collect(1:9)
Jnd = collect(1:9)
V = fill(1,9)
F = sparse(Ind, Jnd, V, 9, 12)

#A
Ind = collect(1:9)
Jnd = [fill(10, 3); fill(11, 3); fill(12, 3)]
V = [1; x[16]; x[17]; 1; x[18]; x[19]; 1; x[20]; x[21]]
A = sparse(Ind, Jnd, V, 12, 12)

start_val = vcat(
    fill(1, 10),
    [1, 1, 0.5, 0.5, 0.5],
    fill(0.5, 6),
    fill(1, 9))

new_val = fill(1.0, 31)

##Symbolic

implysym_test = ImplySymbolic(A, S, F, x, start_val)

implysym_test(start_val)

@test implysym_test.imp_cov == imp_cov_start

implysym_test(new_val)

@test implysym_test.imp_cov == imp_cov_new

##Sparse
implyspa_test = ImplySparse(A, S, F, x, start_val)

implyspa_test(start_val)

@test implyspa_test.imp_cov == imp_cov_start

implyspa_test(new_val)

@test implyspa_test.imp_cov == imp_cov_new
