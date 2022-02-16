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
#x = [x...]

#F
Ind = collect(1:nobs)
Jnd = collect(1:nobs)
V = fill(1,nobs)
F = sparse(Ind, Jnd, V, nobs, nnod)

#A
Ind = collect(1:nobs)
Jnd = vcat([fill(nobs+i, nitem) for i in 1:nfact]...)
V = [x...][1:nobs]
A = sparse(Ind, Jnd, V, nnod, nnod)

#S
Ind = collect(1:nnod)
Jnd = collect(1:nnod)
V = [[x...][nobs+1:2nobs]; fill(1.0, nfact)]
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
Σ_function = eval(Symbolics.build_function(Σ_symbolic, x)[2])
# Σ = zeros(size(Σ_symbolic))

# ∇Σ
∇Σ_symbolic = Symbolics.sparsejacobian(vec(Σ_symbolic), [x...])
∇Σ_function = eval(Symbolics.build_function(∇Σ_symbolic, x)[2])

precompile(∇Σ_function, (typeof(∇pre),Vector{Float64}))

pre = zeros(200, 200)
randpar = rand(410)

constr = findnz(∇Σ_symbolic)
∇pre = sparse(constr[1], constr[2], fill(1.0, nnz(∇Σ_symbolic)))

∇Σ_function(∇pre, randpar)

## split compilation
function eval_build(ex, x)
    eval(build_function(ex, x)[2])
end

function compose(fun1, fun2)
    function(pre, par)
        fun1(pre, par)
        fun2(pre, par)
        nothing
    end
end

equalind(length,n) = [i:min(i+n-1, length) for i in 1:n:length]

function get_sparse_batch(M, ind)
    B = copy(M)
    temp = B.nzval[ind]
    B.nzval .= 0
    B.nzval[ind] .= temp
    return B
end

function batched_build_fun(sym, batch, par)
    ind = equalind(nnz(sym), batch)
    batched = [get_sparse_batch(sym, i) for i in ind]
    broadcast(batched -> eval_build(batched, par), batched)
end

function reduce_funs(funs)
    reduce((f1, f2) -> compose(f1, f2), funs)
end

function batched_reduce_funs(funs, batch)
    ind = equalind(length(funs), batch)
    batched = [funs[i] for i in ind]
    reduce_funs(inline_reduce_funs.(batched))
end


batched_build_fun(∇Σ_symbolic, 1000, x)

ind = equalind(nnz(∇Σ_symbolic), batch)
batched = [get_sparse_batch(sym, i) for i in ind]



@variables x[1:Int(1e6)]

xs = [x...]
xs = xs + xs .+ 5

# have to figure out ideal batch size
funs = batched_build_fun(xs, 250)

#fun = reduce_funs(funs)
# have to figure out ideal batch size
fun2 = batched_reduce_funs(funs, 5)

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

######### parsing
using Symbolics, SparseArrays

lat_vars = "f".*string.(1:3)

obs_vars = "x".*string.(1:9)

model_free = """
f1 =~ x1 + x2 + x3
f2 =~ x4 + x5 + x6
f3 =~ x7 + x8 + x9
f1 ~~ f2
f3 ~ f1
"""

model_fixed = """
f1 =~ 1*x1 + x2 + x3
f2 =~ 1*x4 + x5 + x6
f3 =~ 1*x7 + x8 + x9
f1 ~~ 0.5*f2
f3 ~ f1
"""

model_equal = """
f1 =~ 1*x1 + x2 + x3
f2 =~ 1*x4 + a*x5 + x6
f3 =~ 1*x7 + a*x8 + x9
f1 ~~ 0.5*f2
f3 ~ f1
"""

# functions

function get_parameter_type(string)
    if occursin("~~", string)
        return "~~" 
    elseif occursin("=~", string)
        return "=~"
    elseif occursin("~", string)
        return "~" 
    else
        return nothing
    end
end

function strip_nopar(string_vec)
    parameter_type = get_parameter_type.(string_vec)
    is_par = .!isnothing.(parameter_type)

    string_vec = string_vec[is_par]
    parameter_type = parameter_type[is_par]

    return (string_vec, parameter_type)
end

function expand_model_line(string, parameter_type)
    from, to = split(string, parameter_type)
    to = split(to, "+")

    from = remove_all_whitespace(from)
    to = last.(split.(remove_all_whitespace.(to), "*"))

    free = check_free.(to)
    value_fixed = get_fixed_value.(to, free)
    label = get_label.(to, free)

    from = fill(from, size(to, 1))

    if parameter_type == "=~"
        from, to = copy(to), copy(from)
        parameter_type = "~"
    end

    parameter_type = fill(parameter_type, size(to, 1))

    return from, parameter_type, to, free, value_fixed, label
end

remove_all_whitespace(string) = replace(string, r"\s" => "")

function get_partable(model_vec, parameter_type_in)

    from = Vector{String}()
    to = Vector{String}()
    parameter_type_out = Vector{String}()
    free = Vector{Bool}()
    value_fixed = Vector{Float64}()
    label = Vector{String}()

    for (model_line, parameter_type) in zip(model_vec, parameter_type_in)

        from_new, parameter_type_new, to_new, free_new, value_fixed_new, label_new = 
            expand_model_line(model_line, parameter_type)

        from = vcat(from, from_new)
        parameter_type_out = vcat(parameter_type_out, parameter_type_new)
        to = vcat(to, to_new)
        free = vcat(free, free_new)
        label = vcat(label, label_new)
        value_fixed = vcat(value_fixed, value_fixed_new)

    end

    estimate = copy(value_fixed)

    return from, parameter_type_out, to, free, value_fixed, label, estimate
end

function check_free(to)
    if !occursin("*", to) 
        return true
    else
        fact = split(to, "*")[1]
        fact = remove_all_whitespace(fact)
        if check_str_number(fact)
            return false
        else
            return true
        end
    end
end

function get_label(to, free)
    if !occursin("*", to) || !free
        return ""
    else
        label = split(to, "*")[1]
        return label
    end
end

function get_fixed_value(to, free)
    if free
        return 0.0
    else
        fact = split(to, "*")[1]
        fact = remove_all_whitespace(fact)
        fact = parse(Float64, fact)
        return fact
    end
end

function check_str_number(string)
    return tryparse(Float64, string) !== nothing
end

function parse_sem(model)
    model_vec = split(model, "\n")
    model_vec, parameter_type = strip_nopar(model_vec)
    return get_partable(model_vec, parameter_type)
end

mutable struct ParameterTable
    latent_vars
    observed_vars
    from
    parameter_type
    to
    free
    value_fixed
    label
    start
    estimate
end

Base.getindex(partable::ParameterTable, i::Int) =
    (partable.from[i], 
    partable.parameter_type[i], 
    partable.to[i], 
    partable.free[i], 
    partable.value_fixed[i], 
    partable.label[i])

Base.length(partable::ParameterTable) = length(partable.from)


function get_RAM(partable, parname; to_sparse = true)
    n_labels_unique = size(unique(partable.label), 1) - 1
    n_labels = sum(.!(partable.label .== ""))
    n_parameters = sum(partable.free) - n_labels + n_labels_unique

    parameters = (Symbolics.@variables $parname[1:n_parameters])[1]

    n_observed = size(partable.observed_vars, 1)
    n_latent = size(partable.latent_vars, 1)
    n_node = n_observed + n_latent

    A = zeros(Num, n_node, n_node)
    S = zeros(Num, n_node, n_node)
    F = zeros(Num, n_observed, n_node)
    F[LinearAlgebra.diagind(F)] .= 1.0

    positions = Dict(zip([partable.observed_vars; partable.latent_vars], collect(1:n_observed+n_latent)))
    
    # fill Matrices
    known_labels = Dict{String, Int64}()
    par_ind = 1

    for i in 1:length(partable)

        from, parameter_type, to, free, value_fixed, label = partable[i]

        row_ind = positions[from]
        col_ind = positions[to]

        if !free
            if parameter_type == "~"
                A[row_ind, col_ind] = value_fixed
            else
                S[row_ind, col_ind] = value_fixed
                S[col_ind, row_ind] = value_fixed
            end
        else
            if label == ""
                set_RAM_index(A, S, parameter_type, row_ind, col_ind, parameters[par_ind])
                par_ind += 1
            else
                if haskey(known_labels, label)
                    known_ind = known_labels["label"]
                    set_RAM_index(A, S, parameter_type, row_ind, col_ind, parameters[known_ind])
                else
                    known_labels[label] = par_ind
                    set_RAM_index(A, S, parameter_type, row_ind, col_ind, parameters[par_ind])
                    par_ind +=1
                end
            end
        end

    end

    if to_sparse
        A = sparse(A)
        S = sparse(S)
        F = sparse(F)
    end

    return A, S, F, parameters
end

function set_RAM_index(A, S, parameter_type, row_ind, col_ind, parameter)
    if parameter_type == "~"
        A[row_ind, col_ind] = parameter
    else
        S[row_ind, col_ind] = parameter
        S[col_ind, row_ind] = parameter
    end
end



# do it
my_partable = ParameterTable(lat_vars, obs_vars, parse_sem(model_equal)...)

A, S, F, parameters = get_RAM(my_partable, :x)

using DataFrames

DataFrame(partable::ParameterTable) = 
    DataFrame([partable.from, partable.parameter_type, partable.to, partable.start], ["from", "op", "to", "start"])

show(io::IO, partable::ParameterTable) = 
    show(io, DataFrame([partable.from, partable.parameter_type, partable.to], ["from", "op", "to"]))

sem(model)

sem_lavaan()


### type stability for MG Sems
function compose(fun1, fun2)
    function(x)
        F = fun1(x) + fun2(x)
        return F
    end
end

function reduce_funs(funs)
    reduce((f1, f2) -> compose(f1, f2), funs)
end

funs = [sin; cos]

function testfun(a)
    a = a + a^2
    return a
end

funs = [sin; cos; testfun]

f_composed = reduce_funs(funs)

f_composed(2)

@code_warntype f_composed(2)


using SEM, CSV, DataFrames, SparseArrays, Symbolics, LineSearches, Optim, Test, LinearAlgebra

############################################################################
### observed data
############################################################################

dat = DataFrame(CSV.File("examples/data/data_multigroup.csv"))
par_ml = DataFrame(CSV.File("examples/data/par_multigroup_ml.csv"))
par_ls = DataFrame(CSV.File("examples/data/par_multigroup_ls.csv"))

par_ml = filter(row -> (row.free != 0)&(row.op != "~1"), par_ml)
par_ls = filter(row -> (row.free != 0)&(row.op != "~1"), par_ls)

dat = 
    select(
        dat,
        [:school, :x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9])

dat_g1 = select(filter(row -> row.school == "Pasteur", dat), Not(:school))
dat_g2 = select(filter(row -> row.school == "Grant-White", dat), Not(:school))

dat = select(dat, Not(:school))

# observed
semobserved_g1 = SemObsCommon(data = Matrix{Float64}(dat_g1))
semobserved_g2 = SemObsCommon(data = Matrix{Float64}(dat_g2))

############################################################################
### define models
############################################################################

@Symbolics.variables x[1:36]
#x = [x...]

F = zeros(9, 12)
F[diagind(F)] .= 1.0

A = zeros(Num, 12, 12)
A[1, 10] = 1.0; A[4, 11] = 1.0; A[7, 12] = 1.0
A[2:3, 10] .= [x...][16:17]; A[5:6, 11] .= [x...][18:19]; A[8:9, 12] .= [x...][20:21]; 

# group 1
S1 = zeros(Num, 12, 12)
S1[diagind(S1)] .= [x...][1:12]
S1[10, 11] = x[13]; S1[11, 10] = x[13]
S1[10, 12] = x[14]; S1[12, 10] = x[14]
S1[12, 11] = x[15]; S1[11, 12] = x[15]

# group 2
S2 = zeros(Num, 12, 12)
S2[diagind(S2)] .= [x...][22:33]
S2[10, 11] = x[34]; S2[11, 10] = x[34]
S2[10, 12] = x[35]; S2[12, 10] = x[35]
S2[12, 11] = x[36]; S2[11, 12] = x[36]

# S
S1 = sparse(S1)
S2 = sparse(S2)

#F
F = sparse(F)

#A
A = sparse(A)

# diff
diff = 
    SemDiffOptim(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100), 
        #P = 0.5*inv(H0),
        #precondprep = (P, x) -> 0.5*inv(FiniteDiff.finite_difference_hessian(model_ls_ana, x))), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8))

### start values
par_order = [collect(7:21); collect(1:6); collect(28:42)]

####################################################################
# ML estimation
####################################################################

start_val_ml = Vector{Float64}(par_ml.start[par_order])

# loss
loss_ml_g1 = SemLoss((SemML(semobserved_g1, 1.0, similar(start_val_ml)),))
loss_ml_g2 = SemLoss((SemML(semobserved_g2, 1.0, similar(start_val_ml)),))

# imply
imply_ml_g1 = RAMSymbolic(A, S1, F, x, start_val_ml)
imply_ml_g2 = RAMSymbolic(A, S2, F, x, start_val_ml)

# models
model_ml_g1 = Sem(semobserved_g1, imply_ml_g1, loss_ml_g1, SemDiffOptim(nothing, nothing))
model_ml_g2 = Sem(semobserved_g2, imply_ml_g2, loss_ml_g2, SemDiffOptim(nothing, nothing))

model_ml_multigroup = SemEnsemble((model_ml_g1, model_ml_g2), diff, start_val_ml)

############################################################################
### test gradients
############################################################################

using FiniteDiff

@testset "ml_gradients" begin
    grad = similar(start_val_ml)
    grad .= 0.0
    model_ml_multigroup(start_val_ml, 1.0, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ml_multigroup(x, 1.0, nothing, nothing), start_val_ml)
    grad .= 0.0
    model_ml_multigroup(start_val_ml, nothing, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ml_multigroup(x, 1.0, nothing, nothing), start_val_ml)
end

# fit
solution_ml = sem_fit(model_ml_multigroup)
@test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)


mutable struct teststr2{A, B}
    a::A
    b::B
end

inst = teststr2(zeros(10,10), zeros(10,10))


a = rand(10, 10)

b = rand(10, 10)

d = rand(10, 10)

using BenchmarkTools

@benchmark mul!($c, $a, $b)

########################### 

using Pkg

Pkg.activate("test")

using CSV

Pkg.activate(".")

using DataFrames, StructuralEquationModels, Symbolics, 
    LinearAlgebra, SparseArrays, Optim, LineSearches,
    BenchmarkTools

import StructuralEquationModels as SEM

data = DataFrame(CSV.File("benchmark/regsem/data.csv"))

data = select(data, Not(:Column1))

semobserved = SemObsCommon(data = Matrix{Float64}(data))

############################################################################
### define models
############################################################################

include(pwd()*"/src/frontend/parser.jl")

lat_vars = ["f1"]

obs_vars = "x".*string.(1:9)

model = """
f1 =~ 1*x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
x1 ~~ x1
x2 ~~ x2
x3 ~~ x3
x4 ~~ x4
x5 ~~ x5
x6 ~~ x6
x7 ~~ x7
x8 ~~ x8
x9 ~~ x9
f1 ~~ f1
"""

# do it
my_partable = ParameterTable(
    lat_vars, 
    obs_vars, 
    parse_sem(model)...)


A, S, F, parameters = get_RAM(my_partable, :x)

pretty_table(Dict(my_partable))

start_val = start_simple(Matrix(A), Matrix(S), Matrix(F), parameters)

imply = RAMSymbolic(A, S, F, parameters, start_val)

semdiff =
    SemDiffOptim(
        BFGS(;
        linesearch = BackTracking(order=3),
        alphaguess = InitialHagerZhang()),
        Optim.Options(
            ;f_tol = 1e-10,
            x_tol = 1.5e-8))

lossfun_ml = SemML(semobserved, length(start_val))
lossfun_c = SemConstant(-(logdet(semobserved.obs_cov) + 18), length(start_val))

loss_ml = SemLoss(
    (
        lossfun_ml,
    )
)

model_ml = Sem(semobserved, imply, loss_ml, semdiff)

solution = sem_fit(model_ml)

imply = RAMSymbolic(A, S, F, parameters, start_val)

A, S, F, parameters = get_RAM(my_partable, :x)

A = Matrix(A)
S = Matrix(S)
F = Matrix(F)
M = [parameters[1], parameters[2], 1.0, 0.0]

A_indices = []
S_indices = []

for parameter in parameters

    A_indices_par = []
    S_indices_par = []

    for index in eachindex(A)
        if isequal(parameter, A[index])
            push!(A_indices_par, index)
        end
    end

    for index in eachindex(S)
        if isequal(parameter, S[index])
            push!(S_indices_par, index)
        end
    end

    push!(A_indices, A_indices_par)
    push!(S_indices, S_indices_par)
end

A_indices = [convert(Vector{Int}, indices) for indices in A_indices]
S_indices = [convert(Vector{Int}, indices) for indices in S_indices]

A_pre = zeros(size(A)...)
S_pre = zeros(size(S)...)

for index in eachindex(A)
    δ = tryparse(Float64, string(A[index]))
    if !iszero(A[index]) & (δ !== nothing)
        A_pre[index] = -δ
    end
end

for index in eachindex(S)
    δ = tryparse(Float64, string(S[index]))
    if !iszero(S[index]) & (δ !== nothing)
        S_pre[index] = -δ
    end
end

n_par = length(parameters)
randpar = rand(n_par)

# fill the arrays
function fill_A_S(A_pre, S_pre, A_indices, S_indices, parameters)
    for (iA, iS, par) in zip(A_indices, S_indices, parameters)
        for index_A in iA
            A_pre[index_A] = -par
        end
        for index_S in iS
            S_pre[index_S] = -par
        end
    end
end

# @benchmark fill_A_S($A_pre, $S_pre, $A_indices, $S_indices, $randpar)

fill_A_S(A_pre, S_pre, A_indices, S_indices, randpar)

acyclic = isone(det(I-A_pre))

if iszero(A_pre[.!tril(ones(Bool,10,10))])
    A_pre = LowerTriangular(A_pre)
elseif iszero(A_pre[.!tril(ones(Bool,10,10))'])
    A_pre = UpperTriangular(A_pre)
end

inviat = LowerTriangular(permutedims(invia))
Ft = permutedims(F)

function Σ_RAM!(Σ, At, Ft, S, pre1, pre2)
    ldiv!(pre1, At, Ft)
    mul!(pre2, S, pre1)
    mul!(Σ, pre1', pre2)
end

Σₜ = F*inv(invia)*S_pre*inv(invia')*F'

Σ = zeros(9, 9)
pre1 = zeros(10, 9)
pre2 = zeros(10, 9)

Σ_RAM!(Σ, inviat, Ft, S_pre, pre1, pre2)

Σ ≈ Σₜ

@benchmark F*inv(invia)*S_pre*inv(invia')*F'

@benchmark Σ_RAM!(Σ, inviat, Ft, S_pre, pre1, pre2)

using LinearAlgebra, MKL, Random, SparseArrays, BenchmarkTools

A = I+sprand(100, 100, 0.1)

B = Matrix(copy(A))

C = I

D = rand(100, 100)

@benchmark B\I

@benchmark B\D

######################
M_indices = []


for parameter in parameters

    M_indices_par = []

    for index in eachindex(M)
        if isequal(parameter, M[index])
            push!(M_indices_par, index)
        end
    end

    push!(M_indices, M_indices_par)

end

M_indices = [convert(Vector{Int}, indices) for indices in M_indices]

M_pre = zeros(size(M)...)

for index in eachindex(M)
    δ = tryparse(Float64, string(M[index]))
    if !iszero(M[index]) & (δ !== nothing)
        M_pre[index] = δ
    end
end

S_ind = [findall(!iszero, S[:, i]) for i in 1:size(S, 2)]

S_ind

grad = zeros(20)
n_par = 20

@benchmark $grad .= transpose(vec($A)'*$S)

using BenchmarkTools, LinearAlgebra, SparseArrays, MKL

function kronecker_I(A, n, pre)
    for i in 1:n
        for j in 1:n
            for k in 1:n
                pre[] = A[i, j]
            end
        end
    end
end

n = 3

A = rand(n, n)

B = one(A)

SB = sparse(B)

@benchmark kron($A, $SB)

@benchmark kron($A, $B)

SC = kron(SB, A)
#SC = kron(A, SB)


pre = copy(SC)

function kronecker_I_M(M, n, pre)
    n2 = n^2
    for i = 1:n
        copyto!(pre.nzval, (i-1)*n2+1, M, 1, n2)
        # pre.nzval[(i-1)*n2+1:i*n2] .= vec(M)
    end
    return nothing
end

kronecker_I_M(A, n, pre)

pre == SC

@benchmark kron(SB, A)

@benchmark kronecker_I_M($A, $n, $pre)

Kₙ = sparse(commutation_matrix(n))

p = 90

∇A = sprand(n^2, p, 0.01)

@benchmark Kₙ*pre*∇A

n_var = 80

a = sprand(100, 100, 0.05)
b = Matrix(a)

@benchmark kron($a, $a)

@benchmark $a*$a

@benchmark kron($b, $b)

@benchmark $b*$b

FIA = rand()

A = [0 0 0 0 0 0 1 0 0
    0 0 0 0 0 0 1 0 0
    0 0 0 0 0 0 1 0 0
    0 0 0 0 0 0 0 1 0
    0 0 0 0 0 0 0 1 0
    0 0 0 0 0 0 0 1 0
    0 0 0 0 0 0 0 0 1
    0 0 0 0 0 0 0 0 1
    0 0 0 0 0 0 0 0 0]

A = I-A

A_inv = inv(A)

sparse(A_inv)






using Symbolics, SparseArrays, LinearAlgebra

@variables ω[1:3], ω_12, λ_13, λ_23

Ω = [ω[1]   ω_12    0
    ω_12    ω[2]    0
    0       0       ω[3]]

Λ = [0      0   0
    0       0   0
    λ_13    λ_23 0]

Ω = sparse(Ω)

Λ = sparse(Λ)

Σ = (I + Λ)*Ω*permutedims(I + Λ)

Σ[1,3]

using StructuralEquationModels


function myf(a::Array{b}) where {b <: Float64}
    return a
end