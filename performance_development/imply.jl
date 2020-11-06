using ModelingToolkit, LinearAlgebra,
        SparseArrays, StaticArrays, BenchmarkTools

start_val_ind = vcat(fill(1, 10), [0.5, 0.5, 1, 0.5, 1], fill(0.5, 6), fill(1, 9))
start_val_sym = vcat(fill(1, 10), [1, 1, 0.5, 0.5, 0.5], fill(0.5, 6), fill(1, 9))

## full symbolic computation
@variables x[1:30]

S =[x[1] 0 0 0 0 0 0 0 0 0 0 0.0
    0 x[2] 0 0 0 0 0 0 0 0 0 0
    0 0 x[3] 0 0 0 0 0 0 0 0 0
    0 0 0 x[4] 0 0 0 0 0 0 0 0
    0 0 0 0 x[5] 0 0 0 0 0 0 0
    0 0 0 0 0 x[6] 0 0 0 0 0 0
    0 0 0 0 0 0 x[7] 0 0 0 0 0
    0 0 0 0 0 0 0 x[8] 0 0 0 0
    0 0 0 0 0 0 0 0 x[9] 0 0 0
    0 0 0 0 0 0 0 0 0    x[10] x[13] x[14]
    0 0 0 0 0 0 0 0 0    x[13] x[11] x[15]
    0 0 0 0 0 0 0 0 0    x[14] x[15] x[12]]

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
    0  0  0  0  0  0  0  0  0  x[16] 0     0
    0  0  0  0  0  0  0  0  0  x[17] 0     0
    0  0  0  0  0  0  0  0  0  0     1     0
    0  0  0  0  0  0  0  0  0  0     x[18] 0
    0  0  0  0  0  0  0  0  0  0     x[19] 0
    0  0  0  0  0  0  0  0  0  0     0     1
    0  0  0  0  0  0  0  0  0  0     0     x[20]
    0  0  0  0  0  0  0  0  0  0     0     x[21]
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0]



M = [x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], 0, 0, 0]

# invia = simplify.(simplify.(inv(1.0I-A)))
# covmat_sym =
#     simplify.(
#         simplify.(
#             simplify.(
#                 simplify.(
#                     F*invia
#                 )*
#             S)*
#         permutedims(invia))*
#     permutedims(F))

covmat_sym2 = simplify.(F*simplify.(1.0I+A+A^2+A^3)*S*
             permutedims(simplify.(1.0I+A+A^2+A^3))*permutedims(F))
#
# test = simplify.(A^4)

f1, f2 = eval.(ModelingToolkit.build_function(covmat_sym2, x))

fA = eval.(ModelingToolkit.build_function(A, x))[1]
fS = eval.(ModelingToolkit.build_function(S, x))[1]
fA = eval.(ModelingToolkit.build_function(A, x))[1]

fA(start_val)
fS(start_val)

A1 = f1(start_val_sym)

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

myI = sparse(collect(1:12), collect(1:12), fill(1, 12))

nnz(A^3)

invia = I+A+A^2+A^3
covmat_sym2 = F*invia*S*
             permutedims(invia)*permutedims(F)

covmat_sym2 = simplify.(covmat_sym2)

det(covmat_sym2)

covmat_sym2 = Array(covmat_sym2)

f1, f2 = eval.(ModelingToolkit.build_function(covmat_sym2, x))

Array(f1(start_val_sym)) .== imp_cov
#M = [x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], 0, 0, 0]

struct ram_sym{F <: Any, A <: AbstractArray}
    imp_fun::F
    imp_cov::A
end

function ram_sym(A::Spa1, S::Spa2, F::Spa3, parameters, start_val) where {
    Spa1 <: SparseMatrixCSC, Spa2 <: SparseMatrixCSC, Spa3 <: SparseMatrixCSC}
    invia = I + A
    next_term = A^2
    while nnz(next_term) != 0
        invia += next_term
        next_term *= next_term
    end
    imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym .= simplify.(imp_cov_sym)
    imp_fun_, imp_fun = ModelingToolkit.build_function(
    imp_cov_sym, parameters, expression=Val{false})
    imp_cov = imp_fun_(start_val)
    return ram_sym(imp_fun, imp_cov)
end

function(ram::ram_sym)(parameters)
    ram.imp_fun(ram.imp_cov, parameters)
end

ram_sym_test = ram_sym(A, S, F, x, start_val_sym)

myval = fill(1.0, 31)

@benchmark ram_sym_test(myval)

## use ModelingToolkit to parse
struct ram_par{A <: AbstractArray}
    fA#::F
    fS#::F
    F::A
    A::A
    S::A
    imp_cov::A
    invia::A

end

Ind = [collect(1:12); 10; 10; 11; 11; 12; 12]
Jnd = [collect(1:12); 11; 12; 10; 12; 10; 11]
V = [[x[i] for i = 1:14]; x[13]; x[15]; x[14]; x[15]]
S = sparse(Ind, Jnd, V)
S = Array(S)

#F
Ind = collect(1:9)
Jnd = collect(1:9)
V = fill(1,9)
F = sparse(Ind, Jnd, V, 9, 12)
F = convert(Array{Float64}, Array(F))

#A
Ind = collect(1:9)
Jnd = [fill(10, 3); fill(11, 3); fill(12, 3)]
V = [1; x[16]; x[17]; 1; x[18]; x[19]; 1; x[20]; x[21]]
A = sparse(Ind, Jnd, V, 12, 12)
A = Array(A)

fA = eval.(ModelingToolkit.build_function(A, x))[2]
fS = eval.(ModelingToolkit.build_function(S, x))[2]

ram_par_test = ram_par(fA, fS, F, zeros(12, 12), zeros(12,12), zeros(9,9))

function (ram::ram_par)(par)
    ram.fA(ram.A, par)
    ram.fS(ram.S, par)
    ram.invia .= LinearAlgebra.inv!(factorize(I - ram.A))
    ram.imp_cov .= ram.F*
    LinearAlgebra.inv!(factorize(I - ram.A))*ram.S*transpose(inv(I - ram.A))*transpose(ram.F)
end


## Test
@benchmark ram_ind_test(start_val_ind)


all_cov = ones(9, 9)

@benchmark f2(all_cov, start_val_sym)

@benchmark ram_par_test(start_val_sym)



testmat = I - ram_par_test.A

m1, m2, m3, m4 = rand(10, 10), rand(10,10), rand(10,10), rand(10,10)

@benchmark myf(testmat)

@benchmark myf2(testmat)

function myf(mat)
    mat .= inv(mat)
end

function myf2(mat)
    mat .= LinearAlgebra.inv!(factorize(mat))
end

function mymul1(m1, m2)
    m1 .= m1*m2
end

function mymul2(m1, m2)
    m1 .= m1*transpose(m2)
end

function mymul3(m1, m2, m3)
    mul!(m3, m1, m2)
end

function mymul4(m1, m2, m3, m4)
    transpose!(m4, m2)
    mul!(m3, m1, m4)
end

@benchmark mymul1(m1, m2, m3)

@benchmark mymul2(m1, m2, m3)

@benchmark mymul3(m1, m2, m3)

@benchmark mymul4(m1, m2, m3)

m1s = convert(MMatrix{10,10}, m1)
m2s = convert(MMatrix{10,10}, m2)
m3s = convert(MMatrix{10,10}, m3)
m4s = convert(MMatrix{10,10}, m4)

@benchmark mymul1(m1s, m2s, m3s)

@benchmark mymul3(m1s, m2s, m3s)

@benchmark myf(m1)
@benchmark myf(m1s)


##
#A
@variables x[1:200]

Ind = [collect(1:30); 65; 66; collect(31:60); 20; 88; collect(61:90); 21; 50;
    92; 93; 93]
Jnd = [fill(91, 32); fill(92, 32); fill(93, 32); 91; 91; 92]
V = [x[i] for i = 1:99]
A = sparse(Ind, Jnd, V, 93, 93)

S = sparse(collect(1:93), collect(1:93), [x[i] for i = 100:192])

F = sparse(collect(1:90), collect(1:90), fill(1.0, 90), 90, 93)

start_val = fill(0.5, 192)

@btime imply = ImplySymbolic(A, S, F, x, start_val)

@btime imply(start_val)

imply.imp_cov

using ModelingToolkit

@variables x

A = fill(x, 30, 30)

imp_fun_, imp_fun =
    ModelingToolkit.build_function(
        A,
        x,
        expression=Val{false}
    )

B = rand(30,30)

@btime imp_fun(B, 1)

imply = ImplySymbolic(A, S, F, x, start_val)

@btime imply(start_val)

## parse symbolmatr. to indices # too slow
struct ram_ind{T <: AbstractArray, N <: Union{AbstractArray, Nothing},
    B <: BitArray, B2 <: BitArray, V <: Vector{Int64}, IM <: AbstractArray,
    V2 <: Vector{Any}}
    S::T
    F::T
    A::T
    M::N
    Ind_s::B
    Ind_a::B
    Ind_m::B2
    parind_s::V
    parind_a::V
    parind_m::V
    imp_cov::IM
    names::V2
end


function ram_ind(S, F, A, M, start_values, imp_cov)
    Ind_s = eltype.(S) .== Any
    Ind_a = eltype.(A) .== Any

    sfree = sum(Ind_s)
    afree = sum(Ind_a)

    parsym_a = A[Ind_a]
    parsym_s = S[Ind_s]

    parind_s = zeros(Int64, sfree)
    parind_a = zeros(Int64, afree)

    parind_s[1] = 1

    for i = 2:sfree
        parind_s[i] = maximum(parind_s) + 1
        for j = 1:i
            if parsym_s[i] == parsym_s[j]
                parind_s[i] = parind_s[j]
            end
        end
    end

    parind_a[1] = maximum(parind_s) + 1

    for i = 2:afree
        parind_a[i] = maximum(parind_a) + 1
        for j = 1:i
            if parsym_a[i] == parsym_a[j]
                parind_a[i] = parind_a[j]
            end
        end
    end

    if !isa(M, Nothing)
        Ind_m = eltype.(M) .== Any
        mfree = sum(Ind_m)
        parsym_m = M[Ind_m]
        parind_m = zeros(Int64, mfree)


        parind_m[1] = maximum(parind_a) + 1

        for i = 2:mfree
            parind_m[i] = maximum(parind_m) + 1
            for j = 1:i
                if parsym_m[i] == parsym_m[j]
                    parind_m[i] = parind_m[j]
                end
            end
        end

        M_start = copy(M)
        M_start[Ind_m] .= start_values[parind_m]
        M_start = convert(Array{Float64}, M_start)
    else
        M_start = nothing
        Ind_m = bitrand(2,2)
        parind_m = rand(Int64, 5)
    end

    S_start = copy(S)
    A_start = copy(A)


    S_start[Ind_s] .= start_values[parind_s]
    A_start[Ind_a] .= start_values[parind_a]


    return ram_ind(
        convert(Array{Float64}, S_start),
        convert(Array{Float64}, F),
        convert(Array{Float64}, A_start),
        M_start,
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, S_start),
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, F),
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, A_start),
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, M_start),
        Ind_s, Ind_a, Ind_m, parind_s, parind_a, parind_m, imp_cov,
        [unique(parsym_s); unique(parsym_a)])
end

function (ram::ram_ind{T, N, B, B2,V, IM,V2} where {
    T <: AbstractArray, N <: AbstractArray,
        B <: BitArray, B2 <: BitArray, V <: Vector{Int64}, IM <: AbstractArray,
        V2 <: Vector{Any}
    })(parameters)
    ram.S[ram.Ind_s] .= parameters[ram.parind_s]
    ram.A[ram.Ind_a] .= parameters[ram.parind_a]
    ram.M[ram.Ind_m] .= parameters[ram.parind_m]
end

function (ram::ram_ind{T, N, B, B2,V, IM,V2} where {
    T <: AbstractArray, N <: AbstractArray,
        B <: BitArray, B2 <: BitArray, V <: Vector{Int64}, IM <: AbstractArray,
        V2 <: Vector{Any}
    })(parameters)
    for i = 1:length(ram.Ind_s)
        ram.S[ram.Ind_s[i]]
     .= parameters[ram.parind_s]
    ram.A[ram.Ind_a] .= parameters[ram.parind_a]
    ram.M[ram.Ind_m] .= parameters[ram.parind_m]
end

function (ram::ram_ind{Array{Float64,2},Nothing})(parameters)
    ram.S[ram.Ind_s] .= parameters[ram.parind_s]
    ram.A[ram.Ind_a] .= parameters[ram.parind_a]
end

S =[:x1 0 0 0 0 0 0 0 0 0 0 0.0
    0 :x2 0 0 0 0 0 0 0 0 0 0
    0 0 :x3 0 0 0 0 0 0 0 0 0
    0 0 0 :x4 0 0 0 0 0 0 0 0
    0 0 0 0 :x5 0 0 0 0 0 0 0
    0 0 0 0 0 :x6 0 0 0 0 0 0
    0 0 0 0 0 0 :x7 0 0 0 0 0
    0 0 0 0 0 0 0 :x8 0 0 0 0
    0 0 0 0 0 0 0 0 :x9 0 0 0
    0 0 0 0 0 0 0 0 0 :x10 :x13 :x14
    0 0 0 0 0 0 0 0 0 :x13 :x11 :x15
    0 0 0 0 0 0 0 0 0 :x14 :x15 :x12]

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
    0  0  0  0  0  0  0  0  0  :x16 0     0
    0  0  0  0  0  0  0  0  0  :x17 0     0
    0  0  0  0  0  0  0  0  0  0     1     0
    0  0  0  0  0  0  0  0  0  0     :x18 0
    0  0  0  0  0  0  0  0  0  0     :x19 0
    0  0  0  0  0  0  0  0  0  0     0     1
    0  0  0  0  0  0  0  0  0  0     0     :x20
    0  0  0  0  0  0  0  0  0  0     0     :x21
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0]

M = [:x22, :x23, :x24, :x25, :x26, :x27, :x28, :x29, :x30, 0, 0, 0]

ram_ind_test = ram_ind(S, F, A, M, start_val,
    zeros(9,9))


## inverse

function f1(mat)
    mat2 = copy(mat)
    mat3 = LinearAlgebra.inv!(mat2)
    return mat3
end

function f2(mat, pre)
    mat2 = copy(mat)
    pre .= LinearAlgebra.inv!(mat2)
    return pre
end

function f3(mat, pre)
    mat2 = copy(mat)
    pre .= inv(mat2)
    return pre
end

mat = rand(10,10)

mat = mat'*mat

trueinv = copy(mat)

trueinv = inv(mat)

mat = cholesky(mat)

check = copy(mat)

pre = zeros(10,10)

@benchmark f1($mat)

@benchmark f2($mat, $pre)

@benchmark f3($mat, $pre)

check == mat

Matrix(f1(mat)) ≈ trueinv

f2(mat, pre) ≈ trueinv

mat

check
