using sem, Arrow, ModelingToolkit, LinearAlgebra, 
    SparseArrays, DataFrames, Optim, LineSearches,
    Statistics

cd("test")

## Observed Data
dat = DataFrame(Arrow.Table("comparisons/data_dem.arrow"))
par_ml = DataFrame(Arrow.Table("comparisons/par_dem_ml.arrow"))
par_ls = DataFrame(Arrow.Table("comparisons/par_dem_ls.arrow"))

dat = 
    select(
        dat, 
        [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8])
# observed
semobserved = SemObsCommon(data = Matrix{Float64}(dat))

## Model definition
@ModelingToolkit.variables x[1:31]
@Symbolics.variables x[1:31]

#x = rand(31)

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
    
par_order = [collect(21:34); collect(15:20); 2;3; 5;6;7; collect(9:14)]

start_val_ml = Vector{Float64}(par_ml.start[par_order])
start_val_ls = Vector{Float64}(par_ls.start[par_order])
start_val_snlls = Vector{Float64}(par_ls.start[par_order][21:31])

# loss
loss_ml = Loss([SemML(semobserved, [0.0], similar(start_val_ml))])
loss_ls = Loss([sem.SemWLS(semobserved, [0.0], similar(start_val_ml))])
loss_snlls = Loss([sem.SemSWLS(semobserved, [0.0], similar(start_val_ml))])

# imply
imply_ml = ImplySymbolic(A, S, F, x, start_val_ml)
imply_ls = sem.ImplySymbolicWLS(A, S, F, x, start_val_ls)
imply_snlls = sem.ImplySymbolicSWLS(A, S, F, x[21:31], start_val_snlls)


grad_ls = sem.∇SemWLS(loss_ls.functions[1], size(F, 1))

diff_ana = 
    SemAnalyticDiff(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100), 
        #P = 0.5*inv(H0),
        #precondprep = (P, x) -> 0.5*inv(FiniteDiff.finite_difference_hessian(model_ls_ana, x))), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8),
            (grad_ls,))

model_ls_ana = Sem(semobserved, imply_ls, loss_ls, diff_ana)

grad_ml = sem.∇SemML_2()       

diff_ana_ml = 
    SemAnalyticDiff(
        LBFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# P = one(rand(31, 31))), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8),
            (grad_ml,))

model_ml_ana = Sem(semobserved, imply_ml, loss_ml, diff_ana_ml)

# fit
solution_ana = sem_fit(model_ls_ana)
solution_ana_ml = sem_fit(model_ml_ana)


all(#
            abs.(solution_ana.minimizer .- par_ls.est[par_order]
                ) .< 0.05*abs.(par_ls.est[par_order]))

all(#
            abs.(solution_ana_ml.minimizer .- par_ml.est[par_order]
                ) .< 0.05*abs.(par_ml.est[par_order]))


imply_ls = sem.ImplySymbolicWLS(A, S, F, x, start_val_ls; hessian = true)

# J = rand(66)
# @benchmark imply_ls.hessian_fun(imply_ls.∇²Σ, J, start_val_ls)

hessian_ls = sem.∇²SemWLS(loss_ls.functions[1])

diff_ana_hes = 
    SemAnalyticDiff(
        Newton(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100), 
        #P = 0.5*inv(H0),
        #precondprep = (P, x) -> 0.5*inv(FiniteDiff.finite_difference_hessian(model_ls_ana, x))), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8),
            (grad_ls,),
            (hessian_ls,))

model_ls_hes = Sem(semobserved, imply_ls, loss_ls, diff_ana_hes)

#= randpar = rand(31)
grad = similar(randpar)
H = zeros(31, 31)

model_ls_hes(randpar, grad)
model_ls_hes(randpar, H)

maximum(abs.(grad' - FiniteDiff.finite_difference_jacobian(model_ls_ana, randpar)))
maximum(abs.(H - FiniteDiff.finite_difference_hessian(model_ls_ana, randpar))) =#

solution_hes = sem_fit(model_ls_hes)

all(#
            abs.(solution_hes.minimizer .- par_ls.est[par_order]
                ) .< 0.05*abs.(par_ls.est[par_order]))
################################# hessian #####################################
using BenchmarkTools, FiniteDiff

randpar = rand(31)
H = zeros(31,31)
J = rand(31)
grad = rand(31)

@benchmark model_ls_hes.imply.hessian_fun($H, $J, $randpar)
@benchmark model_ls_hes(randpar, grad)

grad - FiniteDiff.finite_difference_gradient(model_ls_ana, randpar)

@benchmark sem_fit(model_ls_hes)
@benchmark sem_fit(model_ls_ana)

@benchmark model_ls_hes(randpar)
@benchmark model_ls_ana(randpar)

grad = similar(randpar)

@benchmark model_ls_hes(randpar, grad)
@benchmark model_ls_ana(randpar, grad)

H = rand(31,31)

@benchmark model_ls_hes(randpar, H)
@benchmark FiniteDiff.finite_difference_hessian(model_ls_ana, randpar)

H0 = FiniteDiff.finite_difference_hessian(model_ls_ana, randpar)
model_ls_hes(randpar, H) 
H - H0
#H0 = convert(Matrix{Float64}, H0)
#H0 = inv(H0)
@benchmark model_ls_ana.imply.∇Σ'*diff_ana.functions[1].V*model_ls_ana.imply.∇Σ
@benchmark model_ls_ana.imply.∇Σ'*diff_ana.functions[1].V*model_ls_ana.imply.∇Σ

#function mypre_H0(x)
#    return H0
#end

model_ls_ana(randpar)
gr = similar(randpar)
model_ls_ana(randpar, gr)
initial_P = 2*model_ls_ana.imply.∇Σ'*model_ls_ana.diff.functions[1].V*model_ls_ana.imply.∇Σ

sparse(abs.(initial_P-H0) .> 0.0001)

par = [x[i] for i = 1:31]
invia = sem.neumann_series(A)
imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
imp_cov_sym = Array(imp_cov_sym)
imp_cov_sym = ModelingToolkit.simplify.(imp_cov_sym)
#imp_cov_sym = Symbolics.simplify.(imp_cov_sym)
imp_cov_sym = imp_cov_sym[tril(trues(size(F, 1), size(F, 1)))]
∇Σ_sym = ModelingToolkit.sparsejacobian(imp_cov_sym, par)
∇²Σ_sym = ModelingToolkit.jacobian(vec(permutedims(∇Σ_sym)), x)

@time ∇Σ_sym = Symbolics.sparsejacobian(imp_cov_sym, par)
@time ∇²Σ_sym = Symbolics.jacobian(vec(permutedims(∇Σ_sym)), x)
∇Σ_sym = Array(∇Σ_sym)
H_array = [ModelingToolkit.sparsejacobian(∇Σ_sym[i, :], par) for i = 1:66]
@time H_array = [Symbolics.sparsejacobian(∇Σ_sym[i, :], par) for i = 1:66]

H_array = [ModelingToolkit.sparsehessian(imp_cov_sym[i], par) for i = 1:66]

H_array[2].nzval

function similar_sparse_float(S)
    S_tuple = findnz(S)
    S_tuple_new = (S_tuple[1], S_tuple[2], ones(size(S_tuple[3], 1)))
    new = sparse(S_tuple_new..., size(S)...)
end

H_pre = similar_sparse_float.(H_array)

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
        Jsym,
        par
    )[2])

T = zeros(31,31)
randpar = rand(31)
J = rand(66)

@benchmark hessian_fun(T, J, randpar)

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

