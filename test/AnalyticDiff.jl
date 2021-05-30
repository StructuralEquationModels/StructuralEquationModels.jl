using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    SparseArrays, BenchmarkTools, Optim, LineSearches

## Observed Data
dat = Feather.read("test/comparisons/reg_1.feather")
par = Feather.read("test/comparisons/reg_1_par.feather")
start_lav = Feather.read("test/comparisons/reg_1_start.feather")

semobserved = SemObsCommon(data = Matrix(dat))
rel_tol = 3.1956e-13

diff_fin = 
    SemFiniteDiff(
        LBFGS(
            m = 50,
            alphaguess = InitialHagerZhang(), 
            linesearch = HagerZhang()), 
        Optim.Options(;
            f_tol = 1e-10, 
            x_tol = 1.5e-8))

## Model definition
@ModelingToolkit.variables x[1:243]

#F
Ind = collect(1:25)
Jnd = collect(1:25)
V = fill(1,25)
F = sparse(Ind, Jnd, V, 25, 46)

#A
Ind = [collect(2:25); 5; collect(6:25); fill(27, 19); 1; 4]
Jnd = [fill(26, 24); 27; collect(27:46); collect(28:46); 26; 27]
V =[x[1:64]; 1.0; 1.0]
A = sparse(Ind, Jnd, V, 46, 46)

#S
Ind = [collect(1:6); 26; 27; collect(28:46)]
Jnd = [collect(1:6); 26; 27; collect(28:46)]
V = [x[65:72]; fill(1.0, 19)]
S = sparse(Ind, Jnd, V, 46, 46)
xind = 73
for i = 28:46
    for j = (i+1):46
       S[i,j] = x[xind]
       S[j,i] = x[xind]
       xind += 1
    end
end

par_order = [
    collect(2:6); 
    collect(11:29); 
    8;9; 
    collect(49:67);
    collect(106:124); 
    collect(125:130);
    131; 132;
    collect(133:303)]

start_val = start_lav.est[par_order]

#= start_val = vcat(
    fill(1, 2),
    fill(0.5, 3),
    fill(0.1, 19),
    fill(1,2),
    fill(1,19),
    fill(0,19),
    fill(0.5, 6),
    fill(0.05, 2),
    fill(0, 171)
    ) =#

grad_ml = sem.∇SemML(A, S, F, x, start_val)            
diff_ana = 
    SemAnalyticDiff(
        LBFGS(
            m = 50,
            alphaguess = InitialHagerZhang(), 
            linesearch = HagerZhang()), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8),
            (grad_ml,))    
loss = Loss(
    [SemML(semobserved, [0.0], similar(start_val))])

imply = ImplySymbolic(A, S, F, x, start_val)

model_fin = Sem(semobserved, imply, loss, diff_fin)
solution_fin = sem_fit(model_fin)

model_ana = Sem(semobserved, imply, loss, diff_ana)
@btime solution_ana = sem_fit(model_ana)

@btime model_fin(start_val)
grad = similar(start_val)

using FiniteDiff
@btime FiniteDiff.finite_difference_gradient!(grad, model_fin, start_val)

FiniteDiff.GradientCache(
    c1         :: Union{Nothing,AbstractArray{<:Number}},
    c2         :: Union{Nothing,AbstractArray{<:Number}},
    fx         :: Union{Nothing,<:Number,AbstractArray{<:Number}} = nothing,
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(df),
    inplace    :: Type{Val{T3}} = Val{true})

cache = FiniteDiff.GradientCache(grad, start_val)    
@btime FiniteDiff.finite_difference_gradient!(grad, model_fin, start_val, cache)

FiniteDiff.finite_difference_gradient!(
    df,
    f,
    x,
    fdtype::Type{T1}=Val{:central},
    returntype::Type{T2}=eltype(df),
    inplace::Type{Val{T3}}=Val{true};
    [epsilon_factor])

# Cached
FiniteDiff.finite_difference_gradient!(
    df::AbstractArray{<:Number},
    f,
    x::AbstractArray{<:Number},
    cache::GradientCache;
    [epsilon_factor])    

using FiniteDifferences

@btime FiniteDifferences.grad(central_fdm(2, 1), model_fin, start_val)

function (model::Sem{A, B, C, D} where {A, B, C, D <: SemFiniteDiff})(par::Vector, grad::Vector)
    if length(grad) > 0
        grad .= FiniteDifferences.grad(central_fdm(2, 1), model_fin, par)[1]
    end
    return model(par)
end

@code_warntype model_fin.loss(start_val, model_fin)

using ProfileView

function profile_test(n)
    for i = 1:n
        sem_fit(model_fin)
    end
end

ProfileView.@profview profile_test(1)

@btime model_fin(start_val)

grad = similar(start_val)

@btime model_fin(start_val, grad)

using NLopt
diff_nlopt = SemFiniteDiff(:LD_LBFGS, nothing)
model_nlopt = Sem(semobserved, imply, loss, diff_nlopt)
@btime solution_nlopt = sem.sem_fit_nlopt(model_nlopt, 3.1956e-13)


function profile_test_nlopt(n)
    for i = 1:n
        sem.sem_fit_nlopt(model_nlopt, 3.1956e-13)
    end
end

ProfileView.@profview profile_test_nlopt(10)

all(
    abs.(solution_fin.minimizer .- par.est[par_order]
        ) .< 0.05*abs.(par.est[par_order]))

all(
    abs.(solution_ana.minimizer .- par.est[par_order]
        ) .< 0.05*abs.(par.est[par_order]))        


##### regularized
solutions = []

ml = SemML(semobserved, [0.0], similar(start_val))
con = SemConstant(-(logdet(semobserved.obs_cov) + semobserved.n_man))

for i = 1:50
    ridge = sem.SemRidge(0.01*i, 46:64)
    loss = Loss([ml, con, ridge])
    model_fin = Sem(semobserved, imply, loss, diff_fin)
    push!(solutions, sem_fit(model_fin))
end

getindex.(getfield.(solutions, :minimizer), 46)


maximum(abs.(getfield.(solutions, :minimizer)[50] .- solution_fin.minimizer))

fieldnames(typeof(solutions[1]))

pars = getfield.(solutions, :minimizer)
pars = getindex.(pars, [46:64])

pars = permutedims(hcat(pars...))

#using DataFrames

#pars = DataFrame(pars)

#using Plots

plot(pars)

savefig("pars.pdf")