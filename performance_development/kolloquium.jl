using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff, 
    BenchmarkTools

mat = zeros(10,10)
for i in 1:100 mat[i] = i end

F = zeros(8, 10)
F[diagind(F)] .= 1.0

mat2 = zeros(8,8)
@benchmark for i in 1:8 for j in 1:8 mat2[i, j] = mat[i, j] end end

@variables x[1:32]

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

invia = sem.neumann_series(A)

test = invia*S*permutedims(invia)
@benchmark F*test*permutedims(F)

test = Array(test)

test2 = Array{Num,2}(undef, 30, 30)

@benchmark for i in 1:30 for j in 1:30 test2[i, j] = test[i, j] end end

F = zeros(30, 32)
F[diagind(F)] .= 1.0

imp_cov_sym = F*test*permutedims(F)

@benchmark ModelingToolkit.build_function(
            imp_cov_sym,
            [x...],#, m...], 
            load_t
        )

imp_cov_sym = LowerTriangular(imp_cov_sym)

str_f = ModelingToolkit.build_function(
    imp_cov_sym,
    [x...]#, m...], 
    #load_t
)[2]

write("function.jl", string(str_f))

@benchmark eval(str_f)

imp_cov_sym = simplify.(imp_cov_sym)

function comp_fun(A, S, F)
    nobs = size(F, 1)
    invia = sem.neumann_series(A)
    invia = simplify.(invia)
    
    mat = invia*S*permutedims(invia)
    #mat = simplify.(mat)
    imp_cov_sym = Array{Num,2}(undef, nobs, nobs)
    for i in 1:nobs for j in 1:nobs imp_cov_sym[i, j] = mat[i, j] end end

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym = LowerTriangular(imp_cov_sym)
    for i in 1:nobs 
        for j in 1:i 
            imp_cov_sym[i, j] = 
            ModelingToolkit.simplify(imp_cov_sym[i, j])
        end
    end


    imp_fun =
        eval(ModelingToolkit.build_function(
            imp_cov_sym,
            [x..., m...], 
            load_t
        )[2])



    return imp_fun
end

function comp_fun_sw(A, S, F)
    nobs = size(F, 1)
    invia = sem.neumann_series(A)
    invia = simplify.(invia)
    
    mat = invia*S*permutedims(invia)
    #mat = simplify.(mat)
    imp_cov_sym = Array{Num,2}(undef, nobs, nobs)
    for i in 1:nobs for j in 1:nobs imp_cov_sym[i, j] = mat[i, j] end end

    imp_cov_sym = Array(imp_cov_sym)
    #imp_cov_sym = ModelingToolkit.simplify.(imp_cov_sym)

    imp_fun =
        eval(ModelingToolkit.build_function(
            imp_cov_sym,
            [x..., m...], 
            load_t
        )[2])
    return imp_fun
end


###### larger model ########
## Model definition
@ModelingToolkit.variables x[1:103], m[1:2], load_t[1:100]

F = zeros(100, 102)
F[diagind(F)] .= 1.0
F = sparse(F)

M = zeros(Num, 102, 1)
M .= [zeros(100)..., m[1:2]...]
M = sparse(M)

#S
Ind = [1:102..., 101, 102]; J = [1:102..., 102, 101]; V = [x[1:102]..., x[103], x[103]]
S = sparse(Ind, J, V)

#A
Ind = [1:100..., 1:100...]; J = [fill(101, 100)..., fill(102, 100)...]; V = [ones(100)..., load_t...]
A = sparse(Ind, J, V, 102, 102)  

@btime invia = sem.neumann_series(A)
    
@btime imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)

imp_cov_sym = Array(imp_cov_sym)

@btime ModelingToolkit.simplify.(imp_cov_sym)

@time for i in 1:nobs 
    for j in 1:i 
        imp_cov_sym[i, j] = 
        ModelingToolkit.simplify(imp_cov_sym[i, j])
    end
end

imp_cov_sym = ModelingToolkit.simplify.(imp_cov_sym)

str_f = ModelingToolkit.build_function(
    imp_cov_sym,
    [x..., m...], 
    load_t
)[2]

@btime imp_fun =
    eval(str_f)

@btime imp_fun = comp_fun(A, S, F)
@benchmark imp_fun_sw = comp_fun_sw(A, S, F)

s = rand(105)
defvars = rand(100)

pre = zeros(100, 100)
pre_sw = zeros(100, 100)

imp_fun(pre, s, defvars)
nobs = size(pre, 1)
for i in 1:nobs for j in (i+1):nobs pre[i,j] = pre[j,i] end end
imp_fun_sw(pre_sw, s, defvars)

function imp_fun_w(pre, s, defvars)
    imp_fun(pre, s, defvars)
    #nobs = size(pre, 1)
    #@inbounds for i in 1:nobs for j in (i+1):nobs pre[i,j] = pre[j,i] end end
    sym = Symmetric(pre, :L)
    return sym
end
sym = imp_fun_w(pre, s, defvars)

#S
Ind = [1:102..., 101, 102]; J = [1:102..., 102, 101]; V = [s[1:102]..., s[103], s[103]]
S_real = sparse(Ind, J, V)

#A
Ind = [1:100..., 1:100...]; J = [fill(101, 100)..., fill(102, 100)...]; V = [ones(100)..., defvars...]
A_real = sparse(Ind, J, V, 102, 102)

pre_sum = zeros(102,102)
pre_mul1 = zeros(102,102)
pre_mul2 = zeros(102,102)

#A_real^2

function myimp(A, S, pre_sum, pre_mul1, pre_mul2)
    pre_sum = I + A
    mul!(pre_mul1, pre_sum, Symmetric(S))
    mul!(pre_mul2, pre_mul1, transpose(pre_sum))
end

myimp(A_real, S_real, pre_sum, pre_mul1, pre_mul2)

#I + A_real ≈ inv(I-A_real)

cov_check = F*pre_mul2*F'

cov_check ≈ pre
cov_check ≈ pre_sw
sym ≈ cov_check

cholesky!(sym)
nobs = size(pre, 1)
@inbounds for i in 1:nobs for j in (i+1):nobs pre[i,j] = pre[j,i] end end
@benchmark cholesky(sym)


@benchmark myimp($A_real, $S_real, $pre_sum, $pre_mul1, $pre_mul2)

@benchmark imp_fun($pre, $s, $defvars)
@benchmark imp_fun_sw($pre_sw, $s, $defvars)
@benchmark imp_fun_w($pre_sw, $s, $defvars)
@code_warntype imp_fun_w(pre, s, defvars)

######################################################
##
######################################################

## ridge
struct SemRidge{P, W} <: LossFunction
    penalty::P
    which::W
end

function (ridge::SemRidge)(par, model)
    F = ridge.penalty*sum(par[ridge.which].^2)
end


par = rand(10)
myridge = SemRidge(0.5, 1:10)
myridge(par, "mod")

######################################################
## Types and Multiple Dispatch
######################################################

using LinearAlgebra

a = 1.4
b = 1.9

a = rand(10,10)
b = rand(10,10)

typeof(a)

function f(x, y)
    tr(x*inv(y))
end

f(a, b)

b = b*b'

c = cholesky!(b)

f(a, c)
 
@benchmark f(a, b)

@benchmark f(a, c)

@code_lowered f(a, b)
@code_lowered f(a, c)

######################################################
##
######################################################
a = 1.4

typeof(a)

a = rand(10, 10)

struct Point{A, B}
    x::A
    y::B
end

p1 = Point(1.0, 2.0)
p2 = Point(2.0, 3.0)

p1
typeof(p1)

import Base.*, Base.+, Base.transpose, Base.one, Base.zero

*(a::Point, b::Point) = Point(a.x*b.x, a.y*b.y)
+(a::Point, b::Point) = Point(a.x+b.x, b.y+a.y)

p1*p2

M = [Point(1, 2) Point(14, 12)
     Point(1, 3) Point(187, 2)]

M*M     

transpose(M)

transpose(a::Point) = Point(a.y, a.x)

transpose(M)

using SparseArrays

zero(::Type{Point{A, B}}) where {A, B} = Point{A, B}(zero(A), zero(B))
zero(p::Point) = Point(zero(p.x), zero(p.y))

S = zeros(Point{Float64, Float64}, 20, 20)
S[5, 3] = Point(1.4, 2.8)
S[15, 6] = Point(1.7, 9.8)
S[2, 3] = Point(15.6, 14.5)

S2 = sparse(S)

S*S

using BenchmarkTools

@benchmark S*S

@benchmark S2*S2

vec1*vec2

p1 = Point(1.0f0, 2.0f0)
p2 = Point(2.0f0, 3.0f0)

p1*p2

p1 = Point("happy", "garden")
p2 = Point("flower", "sun")

p1*p2

struct Point2{U <: Number, V <: Number}
    x::U
    y::V
end

Point2("happy", "garden")

f(a::Float64) = a^2 + sqrt(a)

f(a::Float32) = true

f(3.0)

f(3.0f0)

@code_lowered f(3.0)

@code_lowered f(3.0f0)
######################################################
##
######################################################

loss = Loss([SemML(semobserved, [0.0], similar(start_val))])

diff_ana = SemAnalyticDiff(LBFGS(), Optim.Options(;show_trace = true),
            A, S, F, x, start_val)

imply = ImplySymbolic(A, S, F, θ, start_val)

imply(start_val)

imply_alloc = sem.ImplySymbolicAlloc(A, S, F, x, start_val)
imply_forward = sem.ImplySymbolicForward(A, S, F, x, start_val)

model_fin = Sem(semobserved, imply, loss, diff_fin)
model_rev = Sem(semobserved, imply_alloc, loss, diff_rev)
model_for = Sem(semobserved, imply_alloc, loss, diff_for)
model_for2 = Sem(semobserved, imply_forward, loss, diff_for)
model_ana = Sem(semobserved, imply, loss, diff_ana)

#Zygote.@nograd isposdef
#Zygote.@nograd Symmetric

@btime diff_fin = SemFiniteDiff("a", "b")

diff_fin = SemFiniteDiff(BFGS(), Optim.Options())

function loopsem(k, sem, semobserved, imply, loss, diff_fin)
    A = Vector{Any}(undef, k)
    for i in 1:k
        A[i] = deepcopy(Sem(semobserved, imply, loss, diff_fin))
    end
    return A
end

@benchmark loopsem(1000, model_fin, semobserved, imply, loss, diff_fin)

testmat = copy(test[1].imply.imp_cov)

test[1].imply.imp_cov .= zeros(11,11)

test[2].imply.imp_cov

test[1].diff.algorithm = "c"

model_fin

ForwardDiff.gradient(model_for, start_val)
FiniteDiff.finite_difference_gradient(model_fin, start_val)
Zygote.gradient(model_for, start_val)

grad = copy(start_val)
model_ana(nothing, grad, start_val)
grad ≈ ForwardDiff.gradient(model_for, start_val)

model_for.loss.functions[1].grad

output = copy(start_val)

model_fin(start_val)

typeof(model_fin.imply)

solution_fin = sem_fit(model_fin)
solution_for = sem_fit(model_for)
solution_for2 = sem_fit(model_for2)
solution_rev = sem_fit(model_rev)
solution_ana = sem_fit(model_ana)