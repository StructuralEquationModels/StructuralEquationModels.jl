using Symbolics, SparseArrays, LinearAlgebra, BenchmarkTools

@Symbolics.variables x[1:31]

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

par = [x[i] for i = 1:31]
invia = I + A + A^2 + A^3 + A^4
imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
imp_cov_sym = Array(imp_cov_sym)
imp_cov_sym = Symbolics.simplify.(imp_cov_sym)
imp_cov_sym = imp_cov_sym[tril(trues(size(F, 1), size(F, 1)))]

∇Σ_sym = Symbolics.sparsejacobian(imp_cov_sym, par)
∇²Σ_sym = Symbolics.sparsejacobian(vec(permutedims(∇Σ_sym)), par)
∇Σ_sym = Array(∇Σ_sym)
# H_array = [Symbolics.sparsejacobian(∇Σ_sym[i, :], par) for i = 1:66]
H_array = [Symbolics.sparsehessian(imp_cov_sym[i], par) for i = 1:66]

#= @variables x[1:31]
par = [x[i] for i = 1:31]
testarray = [x[1]; x[1]*x[2]; x[5]*x[9]]

J = Symbolics.sparsejacobian(testarray, par)
J
H = Symbolics.sparsejacobian(J[3, :], par)
H_array[2].nzval =#

#= function similar_sparse_float(S)
    S_tuple = findnz(S)
    S_tuple_new = (S_tuple[1], S_tuple[2], ones(size(S_tuple[3], 1)))
    new = sparse(S_tuple_new..., size(S)...)
end

H_pre = similar_sparse_float.(H_array) =#

nfact = 5
nitem = 30

## Model definition
nobs = nfact*nitem
nnod = nfact+nobs
@variables x[1:Int64(nobs + nobs)]

x = [x[i] for i = 1:size(x, 1)]
#F
Ind = collect(1:nobs)
Jnd = collect(1:nobs)
V = fill(1,nobs)
F = sparse(Ind, Jnd, V, nobs, nnod)

#S
Ind = collect(1:nnod)
Jnd = collect(1:nnod)
V = [x[1:nobs]; fill(1.0, nfact)]
S = sparse(Ind, Jnd, V, nnod, nnod)


#A
Ind = collect(1:nobs)
Jnd = vcat([fill(nobs+i, nitem) for i in 1:nfact]...)
V = x[(nobs+1):(2*nobs)]
A = sparse(Ind, Jnd, V, nnod, nnod)

invia = I + A
imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
imp_cov_sym = Array(imp_cov_sym)
imp_cov_sym = Symbolics.simplify.(imp_cov_sym)
imp_cov_sym = imp_cov_sym[tril(trues(size(F, 1), size(F, 1)))]

∇Σ_sym = Symbolics.sparsejacobian(imp_cov_sym, x)
#∇²Σ_sym = Symbolics.sparsejacobian(vec(permutedims(∇Σ_sym)), par)
∇Σ_sym = Array(∇Σ_sym)
# H_array = [Symbolics.sparsejacobian(∇Σ_sym[i, :], par) for i = 1:66]
@time H_array = [Symbolics.sparsehessian(imp_cov_sym[i], x) for i = 1:size(imp_cov_sym, 1)]

HT = zeros(Num,size(H_array[1])...)
@variables J[1:size(H_array, 1)]

for i in 1:size(H_array, 1)
    HT += J[i]*H_array[i]
end

using StatsBase
nobs = 80
npar = 100
nnd = Int64(nobs*(nobs+1)/2)

testsym = StatsBase.sample(imp_cov_sym, nnd)
testpar = StatsBase.sample(x, npar)

∇testsym = Symbolics.sparsejacobian(testsym, testpar)
∇testsym = Array(∇testsym)
#∇²Σ_sym = ModelingToolkit.jacobian(vec(permutedims(∇Σ_sym)), x)
@time H_array = [Symbolics.sparsejacobian(∇testsym[i, :], testpar) for i = 1:nnd]
# H_array = [sparse(∇²Σ_sym[((i-1)*31+1):i*31, :]) for i = 1:66]

#= jacobian_fun =
    eval(ModelingToolkit.build_function(
        ∇Σ_sym,
        x
    )[2]) =#

hessian_fun =
    eval(ModelingToolkit.build_function(
        ∇²Σ_sym,
        x
    )[2])

H = zeros(2046, 31)
@benchmark hessian_fun(H, randpar)
#H_array = [sparse(H[((i-1)*31+1):i*31, :]) for i = 1:66]

#model_ls_ana(randpar)

J = (-2*(grad_ls.s-model_ls_ana.imply.imp_cov)'*grad_ls.V)'

@variables J[1:nnd]

HT = zeros(Num, npar, npar)
for i in 1:nnd
    HT += J[i]*H_array[i]
end

@variables Jsym[1:66]

HT = zeros(Num, 31,31)
for i in 1:66
    HT += Jsym[i]*H_array[i]
end

HT = simplify.(HT)

hessian_fun =
    eval(ModelingToolkit.build_function(
        HT,
        x,Jsym
    )[2])

T = zeros(31,31)

@benchmark hessian_fun(T, randpar, J)

T

maximum(abs.(H0 - (initial_P + T)))

f(Hsum, x, J)

maximum(abs.(H0 - (initial_P+HT)))



vectorized_hessian_fun =
    eval(ModelingToolkit.build_function(
        H_array,
        x
    )[2])

vectorized_hessian_fun_alloc =
    eval(ModelingToolkit.build_function(
        H_array,
        x
    )[1])


function mypre(x)
    return initial_P
end

function precondprep_WLS(P, x, model)
    model.imply.gradient_fun(model.imply.∇Σ, x)
    H = 2*model.imply.∇Σ'*model.diff.functions[1].V*model.imply.∇Σ
    return H
end

A_ldiv_B!(pgr, P, gr) = copyto!(pgr, P \ gr)
dot(x, P, y) = dot(x, P*y)

D = duplication_matrix(observed.n_man)
S = inv(observed.obs_cov)
S = kron(S,S)
V = 0.5*(D'*S*D)

LinearAlgebra.ldiv!(P::Matrix{Float64}, b::Vector{Float64}) = P \ b
LinearAlgebra.ldiv!(x::Vector{Float64}, P::Matrix{Float64}, b::Vector{Float64}) = copyto!(x, P \ b)

########################## one factor model ###########################

using Symbolics, SparseArrays, LinearAlgebra, BenchmarkTools

nfact = 1
nitem = 5

## Model definition
nobs = nfact*nitem
nnod = nfact+nobs
@variables x[1:Int64(nobs + nobs)]

x = [x[i] for i = 1:size(x, 1)]
#F
Ind = collect(1:nobs)
Jnd = collect(1:nobs)
V = fill(1,nobs)
F = sparse(Ind, Jnd, V, nobs, nnod)

#S
Ind = collect(1:nnod)
Jnd = collect(1:nnod)
V = [x[1:nobs]; fill(1.0, nfact)]
S = sparse(Ind, Jnd, V, nnod, nnod)


#A
Ind = collect(1:nobs)
Jnd = vcat([fill(nobs+i, nitem) for i in 1:nfact]...)
V = x[(nobs+1):(2*nobs)]
A = sparse(Ind, Jnd, V, nnod, nnod)

S[6,6] = x[6]
A[1,6] = 1

par = [x[i] for i = 1:size(x, 1)]
invia = I + A + A^2 + A^3 + A^4
imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
imp_cov_sym = Array(imp_cov_sym)
imp_cov_sym = Symbolics.simplify.(imp_cov_sym)
imp_cov_sym = imp_cov_sym[tril(trues(size(F, 1), size(F, 1)))]
#imp_cov_sym = vec(imp_cov_sym)
∇Σ_sym = Symbolics.sparsejacobian(imp_cov_sym, par)
∇²Σ_sym = Symbolics.sparsejacobian(vec(permutedims(∇Σ_sym)), par)
∇Σ_sym = Array(∇Σ_sym)
# H_array = [Symbolics.sparsejacobian(∇Σ_sym[i, :], par) for i = 1:66]
H_array = [Symbolics.sparsehessian(imp_cov_sym[i], par) for i = 1:size(imp_cov_sym, 1)]



@Symbolics.variables x[1:3, 1:3]

a = reshape([x[i, j] for j in 1:3 for i in 1:3], (3,3))

a = [x[1] x[2]
    x[3] x[4]]

s = inv(a)

@Symbolics.variables λ[1:2], ω[1:4]

S =[ω[1]  0     0     0     
    0     ω[2]  0     0     
    0     0     ω[3]  0     
    0     0     0     ω[4]]

F =[1.0 0 0 0
    0 1 0 0
    0 0 1 0]

A =[0  0  0  1
    0  0  0  λ[1]
    0  0  0  λ[2]
    0  0  0  0]

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)

invia = I + A + A^2
imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
imp_cov_sym = Array(imp_cov_sym)
imp_cov_sym = Symbolics.simplify.(imp_cov_sym)

Σ_inv = inv(imp_cov_sym)

@Symbolics.variables s[1:3, 1:3]

Σ_obs = reshape([s[i, j] for j in 1:3 for i in 1:3], (3,3))

F = log(det(imp_cov_sym)) + tr(Σ_inv*Σ_obs)

F = simplify(F)

