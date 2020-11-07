using sem, Test, ModelingToolkit, LinearAlgebra, SparseArrays

function impfun(x)

    S =[x[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
        0     x[2]  0     0     0     0     0     0     0     0     0     0     0     0
        0     0     x[3]  0     0     0     0     0     0     0     0     0     0     0
        0     0     0     x[4]  0     0     0     x[15] 0     0     0     0     0     0
        0     0     0     0     x[5]  0     x[16] 0     x[17] 0     0     0     0     0
        0     0     0     0     0     x[6]  0     0     0     x[18] 0     0     0     0
        0     0     0     0     x[16] 0     x[7]  0     0     0     x[19] 0     0     0
        0     0     0     x[15] 0     0     0     x[8]  0     0     0     0     0     0
        0     0     0     0     x[17] 0     0     0     x[9]  0     x[20] 0     0     0
        0     0     0     0     0     x[18] 0     0     0     x[10] 0     0     0     0
        0     0     0     0     0     0     x[19] 0     x[20] 0     x[11] 0     0     0
        0     0     0     0     0     0     0     0     0     0     0     x[12] 0     0
        0     0     0     0     0     0     0     0     0     0     0     0     x[13] 0
        0     0     0     0     0     0     0     0     0     0     0     0     0     x[14]]

    F =[1.0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 1 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 1 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 1 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 1 0 0 0]

    A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
        0  0  0  0  0  0  0  0  0  0  0     x[21] 0     0
        0  0  0  0  0  0  0  0  0  0  0     x[22] 0     0
        0  0  0  0  0  0  0  0  0  0  0     0     1     0
        0  0  0  0  0  0  0  0  0  0  0     0     x[23] 0
        0  0  0  0  0  0  0  0  0  0  0     0     x[24] 0
        0  0  0  0  0  0  0  0  0  0  0     0     x[25] 0
        0  0  0  0  0  0  0  0  0  0  0     0     0     1
        0  0  0  0  0  0  0  0  0  0  0     0     0     x[26]
        0  0  0  0  0  0  0  0  0  0  0     0     0     x[27]
        0  0  0  0  0  0  0  0  0  0  0     0     0     x[28]
        0  0  0  0  0  0  0  0  0  0  0     0     0     0
        0  0  0  0  0  0  0  0  0  0  0     x[29] 0     0
        0  0  0  0  0  0  0  0  0  0  0     x[30] x[31] 0]

    impcov = F*inv(I-A)*S*permutedims(inv(I-A))*permutedims(F)

    return impcov
end

@variables x[1:31]

S =[x[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
    0     x[2]  0     0     0     0     0     0     0     0     0     0     0     0
    0     0     x[3]  0     0     0     0     0     0     0     0     0     0     0
    0     0     0     x[4]  0     0     0     x[15] 0     0     0     0     0     0
    0     0     0     0     x[5]  0     x[16] 0     x[17] 0     0     0     0     0
    0     0     0     0     0     x[6]  0     0     0     x[18] 0     0     0     0
    0     0     0     0     x[16] 0     x[7]  0     0     0     x[19] 0     0     0
    0     0     0     x[15] 0     0     0     x[8]  0     0     0     0     0     0
    0     0     0     0     x[17] 0     0     0     x[9]  0     x[20] 0     0     0
    0     0     0     0     0     x[18] 0     0     0     x[10] 0     0     0     0
    0     0     0     0     0     0     x[19] 0     x[20] 0     x[11] 0     0     0
    0     0     0     0     0     0     0     0     0     0     0     x[12] 0     0
    0     0     0     0     0     0     0     0     0     0     0     0     x[13] 0
    0     0     0     0     0     0     0     0     0     0     0     0     0     x[14]]

F =[1.0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 0 0]

A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[21] 0     0
    0  0  0  0  0  0  0  0  0  0  0     x[22] 0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0  0  0     0     x[23] 0
    0  0  0  0  0  0  0  0  0  0  0     0     x[24] 0
    0  0  0  0  0  0  0  0  0  0  0     0     x[25] 0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[26]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[27]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[28]
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[29] 0     0
    0  0  0  0  0  0  0  0  0  0  0     x[30] x[31] 0]

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)

start_val = vcat(
    vec(var(Matrix(three_path_dat), dims = 1))./2,
    fill(0.05, 3),
    fill(0.0, 6),
    fill(1.0, 8),
    fill(0, 3)
    )

new_val = fill(1.0, 31)

new_val_2 = rand(31)

##Symbolic

implysym_test = ImplySymbolic(A, S, F, x, start_val)

implysym_test(start_val)

@test implysym_test.imp_cov == impfun(start_val)

implysym_test(new_val)

@test implysym_test.imp_cov == impfun(new_val)

implysym_test(new_val_2)

@test implysym_test.imp_cov ≈ impfun(new_val_2)

invia = I + A
next_term = A^2

while nnz(next_term) != 0
    global invia += next_term
    global next_term *= A
end

ModelingToolkit.simplify.(invia)

invia2 = I + A + A^2 + A^3

ModelingToolkit.simplify.(invia2)

imp_cov_sym = F*invia2*S*permutedims(invia2)*permutedims(F)

imp_cov_sym = Array(imp_cov_sym)
imp_cov_sym .= ModelingToolkit.simplify.(imp_cov_sym)

imp_fun =
    eval(ModelingToolkit.build_function(
        imp_cov_sym,
        x
    )[2])

mat = rand(11,11)

imp_fun(mat, start_val)

imp_fun(mat, new_val_2)

mat ≈ impfun(new_val_2)

##Sparse
implyspa_test = ImplySparse(A, S, F, x, start_val)

implyspa_test(start_val)

@test implyspa_test.imp_cov == impfun(start_val)

implyspa_test(new_val)

@test implyspa_test.imp_cov .== impfun(new_val)

implyspa_test(new_val_2)

@test implyspa_test.imp_cov .== impfun(new_val_2)
