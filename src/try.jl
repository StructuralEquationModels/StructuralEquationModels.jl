using FiniteDiff

abstract type AbstractSem end

## loss
abstract type SemLossFunction end

struct SemLoss{F <: Tuple}
    functions::F
end

## Diff
abstract type SemDiff end

## Obs
abstract type SemObs end

## Imply
abstract type SemImply end

struct Sem{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff} <: AbstractSem
    observed::O
    imply::I 
    loss::L 
    diff::D
end

function (model::Sem)(par, F, G, H, weight = nothing)
    model.imply(par, F, G, H, model)
    F = model.loss(par, F, G, H, model, weight)
    if !isnothing(weight) F = weight*F end
    return F
end

function (loss::SemLoss)(par, F, G, H, model)
    if !isnothing(F)
        F = zero(eltype(par))
        for lossfun in loss.functions
            F += lossfun(par, F, G, H, model)
        end
        return F
    end
    for lossfun in loss.functions lossfun(par, F, G, H, model) end
end

function (loss::SemLoss)(par, F, G, H, model, weight)
    if !isnothing(F)
        F = zero(eltype(par))
        for lossfun in loss.functions
            F += lossfun(par, F, G, H, model)
        end
        return F
    end
    for lossfun in loss.functions lossfun(par, F, G, H, model) end
end

######################## example ############################

struct myobs <: SemObs end

obsinst = myobs()

struct mydiff <: SemDiff end

diffinst = mydiff()

struct myimply <: SemImply
    Σ
end

struct myml <: SemLossFunction
    Σ
end

struct myhell <: SemLossFunction
    Σ
end

mlinst = myml([0.0])
hellinst = myhell([0.0])
implyinst = myimply([0.0])

modelinst = Sem(obsinst, implyinst, SemLoss((mlinst,hellinst)), diffinst)

function (imply::myimply)(par, F, G, H, model)
    imply.Σ[1] = par^2
end

function (lossfun::myml)(par, F, G, H, model, weight = nothing)
    # do common computations here
    if !isnothing(G)
        if isnothing(weight) G[1] += 4*par else G[1] += weight*4*par end
    end
    # if isnothing(H) end
    if !isnothing(F)
        F = 2*model.imply.Σ[1]
        if !isnothing(weight) F = weight*F end
        return F
    end
end

function (lossfun::myhell)(par, F, G, H, model)
    if !isnothing(G)
        G .+= FiniteDiff.finite_difference_gradient(par -> lossfun(par, model), [par])
    end
    #if isnothing(H) end
    if !isnothing(F)
        F = model.imply.Σ[1]^2
        return F
    end
end

function (lossfun::myhell)(par, model)
    lossfun.Σ[1] = par[1]^2
    return lossfun.Σ[1]^2
end

par = 2.0

grad = [0.0]

2*par^2 + par^4

4*par + 4*par^3

modelinst(par, 0.0, grad, nothing)

grad

using Optim, BenchmarkTools

@benchmark sol_fin = optimize(par -> modelinst(par[1], 0.0, nothing, nothing), [par], LBFGS())

sol_fin.minimizer

@benchmark sol_grad = optimize(Optim.only_fg!((F, G, par) -> modelinst(par[1], F, G, nothing)), [par], LBFGS())

sol_grad.minimizer


################################# system 2 ######################################

################## specify sum in ensemble ##############


struct myobs <: SemObs end

obsinst = myobs()

struct mydiff <: SemDiff end

diffinst = mydiff()

struct myimply <: SemImply
    Σ
end

struct myml <: SemLossFunction end

struct myhell <: SemLossFunction
    Σ
end

mlinst = myml([0.0])
hellinst = myhell([0.0])
implyinst = myimply([0.0])

modelinst = Sem(obsinst, implyinst, SemLoss((mlinst,hellinst)), diffinst)

function (imply::myimply)(par, F, G, H, model)
    imply.Σ[1] = par^2
end

function (lossfun::myml)(par, F, G, H, model, weight = nothing)
    # do common computations here
    if !isnothing(G)
        if isnothing(weight) G[1] += 4*par else G[1] += weight*4*par end
    end
    # if isnothing(H) end
    if !isnothing(F)
        F = 2*model.imply.Σ[1]
        if !isnothing(weight) F = weight*F end
        return F
    end
end

function (lossfun::myhell)(par, F, G, H, model, weight = nothing)
    if !isnothing(G)
        if !isnothing() G .+= FiniteDiff.finite_difference_gradient(par -> lossfun(par, model), [par]) end
    end
    #if isnothing(H) end
    if !isnothing(F)
        F = model.imply.Σ[1]^2
        return F
    end
end

function (lossfun::myhell)(par, model)
    lossfun.Σ[1] = par[1]^2
    return lossfun.Σ[1]^2
end

par = 2.0

grad = [0.0]

2*par^2 + par^4

4*par + 4*par^3

modelinst(par, 0.0, grad, nothing)

grad

using Optim, BenchmarkTools

@benchmark sol_fin = optimize(par -> modelinst(par[1], 0.0, nothing, nothing), [par], LBFGS())

sol_fin.minimizer

@benchmark sol_grad = optimize(Optim.only_fg!((F, G, par) -> modelinst(par[1], F, G, nothing)), [par], LBFGS())

sol_grad.minimizer

function myf1(a)
    if isnothing(a) error("there is a problem") end
end

function myf2(a)
    return a
end

using ForwardDiff, BenchmarkTools

a = 1.0

@benchmark myf1($a)

####################################################################
# SemLab
####################################################################

# Linear Regression

using Distributions, BenchmarkTools, Optim

# Y = Xβ + ε, Y ∼ N(Xβ, σ*I) with σ = Var(ε)

β = rand(10)
X = rand(1000, 10)

σ = 0.5

ε = rand(Normal(0.0, σ), 1000)

Y = X*β + ε

β₀ = (X'*X)\(X'*Y)

(β₀ - β)'

β = rand(10)/100
β[2] = 0.6
β[6] = 0.4

X = rand(1000, 10)

σ = 0.5

ε = rand(Normal(0.0, σ), 1000)

Y = X*β + ε

function rss(β₀, X, Y, N, α)
    Y₀ = X*β₀
    diff = (Y₀ - Y).^2
    RSS = (1/N)*sum(diff) #+ α*sum(β₀.^2)
    return RSS
end

@benchmark rss($β₀, $X, $Y, 10000, 0.005)

start_val = fill(0.5, 10)

result = optimize(β -> rss(β, X, Y, 10000, 0.005), start_val, BFGS(); autodiff = :forward)

result.minimizer


#### small

ind = sample(1:100, 10; replace = false)
a = rand(100, 100)
a_not = a[Not(ind), Not(ind)]
der = rand(100*100,300)
b = rand(size(vec(a_not), 1))'
ind_after = vec(CartesianIndices(a))
ind_after = findall(x -> !(x[1] ∈ ind || x[2] ∈ ind), ind_after)

a_filtered = der[ind_after, :]

function myf_1(der, b, ind)
    res = b*der[ind, :]
    return res
end

@benchmark myf_1($der, $b, $ind_after)
@benchmark $b*$a_filtered

vec_a == vec(a)[Not(findall(x -> (x[1] ∈ rows_del || x[2] ∈ rows_del), ind))]

### Simulation Abhängigkeit der Parameter
using Distributions, Random, Symbolics, SEM, SparseArrays, Optim, Plots

# Y = β₁X1 + β₂X2 + ε

N = 200
β =[0.5; 0.5]
cov_X = [1 0.9
        0.9 1]

ε = rand(Normal(1), N)

X = permutedims(rand(MvNormal(cov_X), N))

Y = X*β + ε
dat = [X Y]

# define SEM
semobserved = SemObsCommon(data = Matrix{Float64}(dat))

@variables x[1:6]

S = [x[1] x[2]  0
     x[2] x[3]  0
     0    0     x[4]]

A = [0    0    0
     0    0    0
     x[5] x[6] 0]

F = [1 0 0
     0 1 0
     0 0 1.0]

start_val_ml = start_simple(A, S, F, x)

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)

# loss
loss_ml = SemLoss((SemML(semobserved, 1.0, similar(start_val_ml)),))

# imply
imply_ml = RAMSymbolic(A, S, F, x, start_val_ml)

# diff
semdiff =
    SemDiffOptim(
        BFGS(),
        Optim.Options(;f_tol = 1e-10, x_tol = 1.5e-8))

# models
model_ml = Sem(semobserved, imply_ml, loss_ml, semdiff)

solution = sem_fit(model_ml)

solution.minimizer

function plotfun(λ₁, λ₂)
    x = solution.minimizer
    x[5] = λ₁
    x[6] = λ₂
    return model_ridge(x, 1.0, nothing, nothing)
end

λ₁ = collect(0.2:0.01:0.8)
λ₂ = collect(0.2:0.01:0.8)

contour(λ₁, λ₂,	plotfun.(λ₁, λ₂'))

savefig("plot.png")


loss_ridge = SemLoss((SemML(semobserved, 1.0, similar(start_val_ml)), SemRidge(.2, 5:6)))

model_ridge = Sem(semobserved, imply_ml, loss_ridge, semdiff)

solution = sem_fit(model_ridge)

solution.minimizer

using LinearAlgebra, MKL, BenchmarkTools

a = rand(200, 200)
a = a*a'
a = cholesky(a)

function testfun(a)
    for i in 1:100000
        b = inv(a)
    end
    b = b
    return b
end

@benchmark inv($a)

############# Symbolics nested

using Symbolics, BenchmarkTools, LinearAlgebra, MKL

a = rand(100, 100)
b = rand(100, 100)
c = rand(100, 100)

@benchmark mul!($c, $a, $b)

Symbolics.@variables x[150, 100]
@variables t, x, y
B = simplify.([t^2 + t + t^2  2t + 4t
                  x + y + y + 2t  x^2 - x^2 + y^2])

Symbolics.build_function(B, x, y, t)[2]

sym_array = [x...]

x = reshape(x, 150, 100)

sympar = copy(x)

x = x+x .+5

pre = zeros(150, 100)

fun_upper = 
    eval(Symbolics.build_function(
        x_upper, sympar;
        skipzeros = true,
        fillzeros = false)[2])

fun_middle = 
        eval(Symbolics.build_function(
            x_middle, sympar;
            skipzeros = true,
            fillzeros = false)[2])

fun_lower = 
    eval(Symbolics.build_function(
        x_lower, sympar;
        skipzeros = true,
        fillzeros = false)[2])

parameters = rand(150,100)
pre .= 0

fun_upper(pre, parameters)
fun_lower(pre, parameters)
fun_middle(pre, parameters)
pre == parameters + parameters .+5


function setzero(array, index)
    newarray = zero(array)
    newarray[:, index] .= array[:, index]
    return newarray
end

arrays = [setzero(x, i) for i in 1:size(x, 2)]

string_fun = Symbolics.build_function(arrays[1], sympar;
    skipzeros = true,
    fillzeros = false)[2]

str2 = Symbolics.build_function(arrays[1], sympar)[2]

str3 = Symbolics.build_function(x, sympar;
    skipzeros = true,
    fillzeros = false)[2]

str5 = Symbolics.build_function(sym_array, x)[2]

eval(Symbolics.build_function(arrays[1], sympar;
        skipzeros = true,
        fillzeros = false)[2])


str4 = ModelingToolkit.build_function(x, sympar)[2]

funs = Vector{Any}(undef, 100)
for i in 1:1
        funs[i] =  
            eval(Symbolics.build_function(arrays[i], sympar;
            skipzeros = true,
            fillzeros = false)[2])
end

function myf(pre, parameters, funs)
    for fun in funs
        fun(pre, parameters)
    end
end

function myf2(pre, parameters)
    fun_upper(pre, parameters)
    fun_middle(pre, parameters)
    fun_lower(pre, parameters)
end

function myf3(pre, parameters, funs)
    map(x -> x(pre, parameters), funs)
end

pre .= 0

myf3(pre, parameters, funs)

@benchmark myf($pre, $parameters, $funs)

pre == parameters

@benchmark myf2($pre, $parameters)

pre == parameters

@benchmark myf3($pre, $parameters, $funs)

function compose(fun1, fun2)
    function(pre, par)
        fun2(pre, par)
        fun1(pre, par)
    end
end

all_funs = reduce(compose, funs)

pre .= 0
all_funs(pre, parameters)

@benchmark all_funs($pre, $parameters)

@code_llvm all_funs(pre, parameters)
@code_llvm myf2(pre, parameters)
@code_llvm myf(pre, parameters, funs)
@code_llvm myf3(pre, parameters, funs)


####### big ∇Σ
using Symbolics, SparseArrays, SEM

nfact = 5
nitem = 40

## Model definition
nobs = nfact*nitem
nnod = nfact+nobs
n_latcov = Int64(nfact*(nfact-1)/2)
npar = 2nobs + n_latcov
Symbolics.@variables x[1:npar]
x = [x...]

#F
Ind = collect(1:nobs)
Jnd = collect(1:nobs)
V = fill(1,nobs)
F = sparse(Ind, Jnd, V, nobs, nnod)

#A
Ind = collect(1:nobs)
Jnd = vcat([fill(nobs+i, nitem) for i in 1:nfact]...)
V = x[1:nobs]
A = sparse(Ind, Jnd, V, nnod, nnod)

#S
Ind = collect(1:nnod)
Jnd = collect(1:nnod)
V = [x[nobs+1:2nobs]; fill(1.0, nfact)]
S = sparse(Ind, Jnd, V, nnod, nnod)
xind = 2nobs+1
for i in nobs+1:nnod
    for j in i+1:nnod
        S[i,j] = x[xind]
        S[j,i] = x[xind]
        xind = xind+1
    end
end

Σ_symbolic = SEM.get_Σ_symbolic_RAM(S, A, F)
    #print(Symbolics.build_function(Σ_symbolic)[2])
Σ_function = eval(Symbolics.build_function(Σ_symbolic, x)[2])
Σ = zeros(size(Σ_symbolic))

# ∇Σ
∇Σ_symbolic = Symbolics.sparsejacobian(vec(Σ_symbolic), [x...])
∇Σ_symbolic
∇Σ_function = eval(Symbolics.build_function(∇Σ_symbolic, x)[2])

pre = zeros(200, 200)
randpar = rand(410)

constr = findnz(∇Σ_symbolic)
∇pre = sparse(constr[1], constr[2], fill(1.0, nnz(∇Σ_symbolic)))


# vec multiply
a = rand(200, 200)
b = rand(200^2, 410)

∇pre_dense = Matrix(∇pre)

@benchmark vec(a)'*∇pre

@benchmark vec(a)'*∇pre_dense

@benchmark vec(a)'*∇pre

if !vech 
    ∇Σ = zeros(size(F, 1)^2, size(par, 1))
else
    ∇Σ = zeros(size(Σ_symbolic, 1), size(par, 1))
end


### Symbolics MWE unexpected behaviour
using Symbolics, SparseArrays

@variables x[1:5]

A = [x[1] 0 x[2]
    x[3]  0 0
    0     0 0]

I + A

A = sparse([1, 1, 2], [1, 3, 1], x[1:3], 3, 3)

A + B

A = sparse([1, 1, 2], [1, 3, 1], Symbolics.scalarize(x[1:3]), 3, 3)

I + A