using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff

## Observed Data
three_path_dat = Feather.read("test/comparisons/three_path_dat.feather")
three_path_par = Feather.read("test/comparisons/three_path_par.feather")

semobserved = SemObsCommon(data = Matrix(three_path_dat))

loss = Loss([SemML(semobserved)])

diff_for = SemForwardDiff(LBFGS(), Optim.Options())


## Model definition
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

fun_A = eval(ModelingToolkit.build_function(A, x)[2])
fun_S = eval(ModelingToolkit.build_function(S, x)[2])


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



imply = ImplySymbolic(A, S, F, x, start_val)

imply_alloc = sem.ImplySymbolicAlloc(A, S, F, x, start_val)

model_for = Sem(semobserved, imply_alloc, loss, diff_for)

ForwardDiff.gradient(model_for, start_val)[1]

D = model_for.observed.obs_cov

size(start_val)

start_val

function mygrad(y, D)

    S =[y[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
        0     y[2]  0     0     0     0     0     0     0     0     0     0     0     0
        0     0     y[3]  0     0     0     0     0     0     0     0     0     0     0
        0     0     0     y[4]  0     0     0     y[15] 0     0     0     0     0     0
        0     0     0     0     y[5]  0     y[16] 0     y[17] 0     0     0     0     0
        0     0     0     0     0     y[6]  0     0     0     y[18] 0     0     0     0
        0     0     0     0     y[16] 0     y[7]  0     0     0     y[19] 0     0     0
        0     0     0     y[15] 0     0     0     y[8]  0     0     0     0     0     0
        0     0     0     0     y[17] 0     0     0     y[9]  0     y[20] 0     0     0
        0     0     0     0     0     y[18] 0     0     0     y[10] 0     0     0     0
        0     0     0     0     0     0     y[19] 0     y[20] 0     y[11] 0     0     0
        0     0     0     0     0     0     0     0     0     0     0     y[12] 0     0
        0     0     0     0     0     0     0     0     0     0     0     0     y[13] 0
        0     0     0     0     0     0     0     0     0     0     0     0     0     y[14]]

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
        0  0  0  0  0  0  0  0  0  0  0     y[21] 0     0
        0  0  0  0  0  0  0  0  0  0  0     y[22] 0     0
        0  0  0  0  0  0  0  0  0  0  0     0     1     0
        0  0  0  0  0  0  0  0  0  0  0     0     y[23] 0
        0  0  0  0  0  0  0  0  0  0  0     0     y[24] 0
        0  0  0  0  0  0  0  0  0  0  0     0     y[25] 0
        0  0  0  0  0  0  0  0  0  0  0     0     0     1
        0  0  0  0  0  0  0  0  0  0  0     0     0     y[26]
        0  0  0  0  0  0  0  0  0  0  0     0     0     y[27]
        0  0  0  0  0  0  0  0  0  0  0     0     0     y[28]
        0  0  0  0  0  0  0  0  0  0  0     0     0     0
        0  0  0  0  0  0  0  0  0  0  0     y[29] 0     0
        0  0  0  0  0  0  0  0  0  0  0     y[30] y[31] 0]

    B = inv(I-A)
    E = B*S*B'
    Σ_inv = inv(F*E*F')

    parnum = size(y, 1)
    d = zeros(parnum)
    gradvec = zeros(parnum)
    for i = 1:parnum
        d .= 0.0
        d[i] = 1.0

        S_der =[d[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
                0     d[2]  0     0     0     0     0     0     0     0     0     0     0     0
                0     0     d[3]  0     0     0     0     0     0     0     0     0     0     0
                0     0     0     d[4]  0     0     0     d[15] 0     0     0     0     0     0
                0     0     0     0     d[5]  0     d[16] 0     d[17] 0     0     0     0     0
                0     0     0     0     0     d[6]  0     0     0     d[18] 0     0     0     0
                0     0     0     0     d[16] 0     d[7]  0     0     0     d[19] 0     0     0
                0     0     0     d[15] 0     0     0     d[8]  0     0     0     0     0     0
                0     0     0     0     d[17] 0     0     0     d[9]  0     d[20] 0     0     0
                0     0     0     0     0     d[18] 0     0     0     d[10] 0     0     0     0
                0     0     0     0     0     0     d[19] 0     d[20] 0     d[11] 0     0     0
                0     0     0     0     0     0     0     0     0     0     0     d[12] 0     0
                0     0     0     0     0     0     0     0     0     0     0     0     d[13] 0
                0     0     0     0     0     0     0     0     0     0     0     0     0     d[14]]

        A_der =[0  0  0  0  0  0  0  0  0  0  0     0     0     0
            0  0  0  0  0  0  0  0  0  0  0     d[21] 0     0
            0  0  0  0  0  0  0  0  0  0  0     d[22] 0     0
            0  0  0  0  0  0  0  0  0  0  0     0     0     0
            0  0  0  0  0  0  0  0  0  0  0     0     d[23] 0
            0  0  0  0  0  0  0  0  0  0  0     0     d[24] 0
            0  0  0  0  0  0  0  0  0  0  0     0     d[25] 0
            0  0  0  0  0  0  0  0  0  0  0     0     0     0
            0  0  0  0  0  0  0  0  0  0  0     0     0     d[26]
            0  0  0  0  0  0  0  0  0  0  0     0     0     d[27]
            0  0  0  0  0  0  0  0  0  0  0     0     0     d[28]
            0  0  0  0  0  0  0  0  0  0  0     0     0     0
            0  0  0  0  0  0  0  0  0  0  0     d[29] 0     0
            0  0  0  0  0  0  0  0  0  0  0     d[30] d[31] 0]

        term = F*B*A_der*E*F'

        Σ_der =  F*B*S_der*B'F' + term + term'

        gradvec[i] = tr(Σ_inv*Σ_der) + tr((-Σ_inv)*Σ_der*Σ_inv*D)
    end

    return gradvec
end

@btime mygrad(start_val, D)

all(mygrad(start_val, D) .≈ ForwardDiff.gradient(model_for, start_val))

@btime model_fin(start_val)

In = [1, 4, 3, 5]; J = [4, 7, 18, 9]; V = [1, 2, -5, 3];

S = sparse(In,J,V)

S[1,4] = 0.0


parnum = size(start_val, 1)
d = zeros(parnum)
gradvec = zeros(parnum)

S_ind_vec = Vector{Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}}()
A_ind_vec = Vector{Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}}()

for i = 1:parnum
    d .= 0.0
    d[i] = 1.0

    S_der =[d[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
            0     d[2]  0     0     0     0     0     0     0     0     0     0     0     0
            0     0     d[3]  0     0     0     0     0     0     0     0     0     0     0
            0     0     0     d[4]  0     0     0     d[15] 0     0     0     0     0     0
            0     0     0     0     d[5]  0     d[16] 0     d[17] 0     0     0     0     0
            0     0     0     0     0     d[6]  0     0     0     d[18] 0     0     0     0
            0     0     0     0     d[16] 0     d[7]  0     0     0     d[19] 0     0     0
            0     0     0     d[15] 0     0     0     d[8]  0     0     0     0     0     0
            0     0     0     0     d[17] 0     0     0     d[9]  0     d[20] 0     0     0
            0     0     0     0     0     d[18] 0     0     0     d[10] 0     0     0     0
            0     0     0     0     0     0     d[19] 0     d[20] 0     d[11] 0     0     0
            0     0     0     0     0     0     0     0     0     0     0     d[12] 0     0
            0     0     0     0     0     0     0     0     0     0     0     0     d[13] 0
            0     0     0     0     0     0     0     0     0     0     0     0     0     d[14]]

    A_der =[0  0  0  0  0  0  0  0  0  0  0     0     0     0
        0  0  0  0  0  0  0  0  0  0  0     d[21] 0     0
        0  0  0  0  0  0  0  0  0  0  0     d[22] 0     0
        0  0  0  0  0  0  0  0  0  0  0     0     0     0
        0  0  0  0  0  0  0  0  0  0  0     0     d[23] 0
        0  0  0  0  0  0  0  0  0  0  0     0     d[24] 0
        0  0  0  0  0  0  0  0  0  0  0     0     d[25] 0
        0  0  0  0  0  0  0  0  0  0  0     0     0     0
        0  0  0  0  0  0  0  0  0  0  0     0     0     d[26]
        0  0  0  0  0  0  0  0  0  0  0     0     0     d[27]
        0  0  0  0  0  0  0  0  0  0  0     0     0     d[28]
        0  0  0  0  0  0  0  0  0  0  0     0     0     0
        0  0  0  0  0  0  0  0  0  0  0     d[29] 0     0
        0  0  0  0  0  0  0  0  0  0  0     d[30] d[31] 0]

    S_der = sparse(S_der)
    A_der = sparse(A_der)

    S_ind = findnz(S_der)
    A_ind = findnz(A_der)

    push!(S_ind_vec, S_ind)
    push!(A_ind_vec, A_ind)
end

test = sparse(S_ind_vec[11]..., matsize...)

matsize = size(A)

function mygrad2(y, D, A, S, F, S_ind_vec, A_ind_vec)

    fun_S(S, y)

    fun_A(A, y)

    B = inv(I-A)
    E = B*S*B'
    Σ_inv = inv(F*E*F')

    parnum = size(y, 1)
    d = zeros(parnum)
    gradvec = zeros(parnum)
    for i = 1:parnum
        S_der = sparse(S_ind_vec[i]..., matsize...)
        A_der = sparse(A_ind_vec[i]..., matsize...)

        term = F*B*A_der*E*F'

        Σ_der =  F*B*S_der*B'F' + term + term'

        gradvec[i] = tr(Σ_inv*Σ_der) + tr((-Σ_inv)*Σ_der*Σ_inv*D)
    end

    return gradvec
end

A_pre = zeros(14,14)
S_pre = zeros(14,14)

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


all(mygrad2(start_val, D, A_pre, S_pre, F, S_ind_vec, A_ind_vec) .≈
    ForwardDiff.gradient(model_for, start_val))

testfun2 = eval(ModelingToolkit.build_function(A, x)[2])

In = [1, 4, 3, 5]; J = [4, 7, 18, 9]; V = [1, 2, -5, 3];
mat = sparse(In,J,V)

testfun2(mat, start_val)

testfun2()

## optim test

function fg!(F,G,x)
  if G != nothing
    G .= nest1()
  end
  if F != nothing
    return res
  end
end

struct nest1{grad}
    G::grad
end

struct nest2{grad}
    G::grad
end

mynest1 = nest1([0.0])

mynest2 = nest2([0.0])

function (nest1::nest1)(F, G, x)
    res = x[1]^2
    if G != nothing
      ForwardDiff.gradient!(nest1.G, x -> x[1]^2, x)
    end
    if F != nothing
      return res
    end
end

function (nest2::nest2)(F, G, x)
    res = x[1]^2
    if G != nothing
      ForwardDiff.gradient!(nest2.G, x -> x[1]^2, x)
    end
    if F != nothing
      return res
    end
end

function nestedfg!(F, G, x)
    if G != nothing
        mynest1(F, G, x)
        mynest2(F, G, x)
        G .= mynest1.G + mynest2.G
    end
    if F != nothing
        return mynest1(F, G, x) + mynest2(F, G, x)
    end
end

Optim.optimize(Optim.only_fg!(nestedfg!), [2.0], Optim.LBFGS())


ForwardDiff.gradient(model_for, fill(0, 31))


function testfun(mat)
    let A, B = mat, mat
    end
end

mat = rand(10,10)

@btime testfun(mat)


## test structs

sem.SemAnalyticDiff(LBFGS(), Optim.Options(), A, S, F, x, start_val)
