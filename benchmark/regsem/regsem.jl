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

#lossfun_c = SemConstant(-(logdet(semobserved.obs_cov) + 18), length(start_val))

loss_ridge = SemLoss(
    (
        lossfun_ml,
        #lossfun_c,
        SemRidge(.02, [1, 2, 6, 7, 8], length(start_val))
    )
)

model_ridge = Sem(semobserved, imply, loss_ridge, semdiff)

solution = sem_fit(model_ridge)

round.(solution.minimizer; digits = 3)'

### benchmark
λ = collect(0:0.001:0.05)

function fit_λ(λ, semobserved, imply, lossfun_ml, lossfun_c)
    loss_ridge = SemLoss(
        (
            lossfun_ml,
            lossfun_c,
            SemRidge(λ, [1, 2, 6, 7, 8], length(start_val))
        )
    )
    model_ridge = Sem(semobserved, imply, loss_ridge, semdiff)
    solution = sem_fit(model_ridge)
    return solution.minimizer
end

@benchmark fit_λ(λ[1], semobserved, imply, lossfun_ml, lossfun_c)
@benchmark solution = sem_fit(model_ridge)

solutions = []

using MKL

@benchmark for λ ∈ λ
        solution = fit_λ(λ, semobserved, imply, lossfun_ml, lossfun_c)
        push!(solutions, solution)
    end setup=(solutions = [])

solutions = hcat(solutions...)

solutions = solutions[[1, 2, 6, 7, 8], :]

using Plots

plot(λ, solutions')

# result: 3500/20 = 175.55 times faster

### multigroup
lossfun_c = SemConstant(-(logdet(semobserved.obs_cov) + 18), length(start_val))

loss_ml = SemLoss(
    (
        lossfun_ml, 
        lossfun_c
    )
)

model_ml = Sem(semobserved, imply, loss_ml, semdiff)

model_ridge_only = Sem(
    semobserved,
    SEM.ImplyEmpty(start_val),
    SemLoss((SemRidge(.02, [1, 2, 6, 7, 8], length(start_val)),)),
    semdiff
)

model_mg = SemEnsemble((model_ml, model_ridge_only), semdiff, start_val; weights = [1.0, 1.0])

solution_mg = sem_fit(model_mg)

round.(solution_mg.minimizer; digits = 2)'

############################################################################
### type stability for multiple loss functions
############################################################################

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


### define models
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

funs = [deepcopy(lossfun_ml) for i in 1:10]

loss_ridge = SemLoss(
    (
        lossfun_c,
        lossfun_c,
        lossfun_c
        #SemRidge(.02, [1, 2, 6, 7, 8], length(start_val))
    )
)

loss_ml = SemLoss(
    (
        lossfun_ml,
    )
)

model_ridge = Sem(semobserved, imply, loss_ridge, semdiff)
model_ml = Sem(semobserved, imply, loss_ml, semdiff)

# using ProfileView

function profile_test(n)
    for i in 1:n
        objective!(model_ridge, start_val)
    end
end

function profile_test_ml(n)
    for i in 1:n
        objective!(model_ml, start_val)
    end
end

using MKL

profile_test(1)
profile_test_ml(1)

# runtime dispatch: red, garbage collection: orange
ProfileView.@profview profile_test(1000000)
ProfileView.@profview profile_test_ml(10000)

@code_warntype objective!(model_ridge, start_val)

using Cthulhu

@descend model_ridge(start_val, 1.0, nothing, nothing)

######################################################
# proximal algorithms
######################################################
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

using ProximalOperators
import StructuralEquationModels: sem_fit, gradient, gradient!, objective, objective!, objective_gradient!

include(pwd()*"/src/optimizer/algorithms/ForwardBackward/ForwardBackwardTools.jl")
include(pwd()*"/src/optimizer/algorithms/ForwardBackward/IterationTools.jl")
include(pwd()*"/src/optimizer/algorithms/ForwardBackward/ProximalGradient.jl")
include(pwd()*"/src/optimizer/algorithms/PANOC/traits.jl")
include(pwd()*"/src/optimizer/algorithms/PANOC/LBFGS.jl")
include(pwd()*"/src/optimizer/algorithms/PANOC/PANOC.jl")

include(pwd()*"/src/diff//Proximal.jl")
include(pwd()*"/src/optimizer/Proximal.jl")

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

start_val = start_fabin3(Matrix(A), Matrix(S), Matrix(F), parameters, semobserved)

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

loss_ridge = SemLoss(
    (
        lossfun_ml,
        lossfun_c,
        SemRidge(.1, [1, 2, 6, 7, 8], length(start_val))
    )
)

loss_ml = SemLoss(
    (
        lossfun_ml,
    )
)

model_ridge = Sem(semobserved, imply, loss_ridge, semdiff)
model_ml = Sem(semobserved, imply, loss_ml, semdiff)


solution_ridge = sem_fit(model_ridge)
solution_ml = sem_fit(model_ml)

# with proximal algorithm

α = zeros(18)
α[[1, 2, 6, 7, 8]] .= 0.2

diff_proximal = SemDiffProximal(
    ForwardBackward(maxit = 1000),
    #ForwardBackward(maxit = 10000),
    (;),
    NormL1(α)
)

model_lasso = Sem(semobserved, imply, loss_ml, diff_proximal)

solution_lasso, iterations = sem_fit(model_lasso)

include(pwd()*"/src/optimizer/algorithms/ProximalGradient/ProximalGradient.jl")

solution, converged, iterations = proximalgradient(model_lasso, model_lasso.diff.g; tolerance = 0.0001, maxit = 10000)

### plot
λ = collect(0:0.01:0.2)

function fit_λ(λ, model, g)
    α = zeros(18)
    α[[1, 2, 6, 7, 8]] .= λ
    g = g(α)
    solution, converged, iterations = proximalgradient(model, g; tolerance = 0.0001, maxit = 10000)
    return solution
end


using MKL


solutions_lasso = []
solutions_ridge = []


@time for λ ∈ λ
    solution = fit_λ(λ, model_lasso, NormL1)
    push!(solutions_lasso, solution)
end

solutions_lasso = hcat(solutions_lasso...)

solutions_lasso = solutions_lasso[[1, 2, 6, 7, 8], :]

using Plots

plot(λ, solutions_lasso'; ylims = [0, Inf])


for λ ∈ λ
    solution = fit_λ(λ, model_lasso, SqrNormL2)
    push!(solutions_ridge, solution)
end

solutions_ridge = hcat(solutions_ridge...)

solutions_ridge = solutions_ridge[[1, 2, 6, 7, 8], :]

using Plots

plot(λ, solutions_ridge'; ylims = [0, Inf])

###############
using Distributions

μ = 0.0
σ = 1.0

NV = Normal(μ, σ)

function weighted_mean(data, true_mean)
    weights = (data .- true_mean).^-2
    weighted_mean = sum(data .* weights)/sum(weights)
end

n_rep = 100

means = zeros(n_rep)
means_weighted = (zeros(n_rep))

for i in 1:100
    data = rand(NV, 100)
    means[i] = mean(data)
    means_weighted[i] = weighted_mean(data, 0.0)
end


mean(means_weighted)

mean(means)

########################################
###########################################################
########################################

using StructuralEquationModels, SparseArrays, Symbolics, CSV, DataFrames, Optim, LineSearches, Plots, BenchmarkTools, FiniteDiff, LinearAlgebra, Distributions
import StructuralEquationModels as sem

dat = DataFrame(CSV.File(pwd()*"/test/examples/data/data_dem.csv"))

dat = 
    select(
        dat, 
        [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8])

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
    0     0     0     0     0     0     0     0     0     0     0     0     0     x[14]];

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
    0 0 0 0 0 0 0 0 0 0 1 0 0 0];

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
    0  0  0  0  0  0  0  0  0  0  0     x[30] x[31] 0];

S, F, A = map(sparse, (S, F, A))

# observed
semobserved = SemObsCommon(data = Matrix{Float64}(dat))
start_val = start_fabin3(A, S, F, x, semobserved)

loss = SemLoss((SemML(semobserved, length(start_val)),
        SemConstant(11*log(2π), length(start_val))))#11 = num observed variables/dim(cov)

imply = RAMSymbolic(A, S, F, x, start_val; hessian = true)
semdiff =
    SemDiffOptim(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100),
        #P = 0.5*inv(H0),
        #precondprep = (P, x) -> 0.5*inv(FiniteDiff.finite_difference_hessian(model_ls_ana, x))),
        Optim.Options(
            ;f_tol = 1e-10,
            x_tol = 1.5e-8))

model = Sem(semobserved, imply, loss, semdiff)

model_mg = SemEnsemble((model,), semdiff, start_val; weights = [size(dat, 1)])

solution = sem_fit(model_mg).minimizer

### plots
function taylor²(x0, f_x0, ∇f_x0, ∇²f_x0, x)
    x1 = x - x0
    f_x0 + #dot(∇f_x0, x1) + 
        0.5*x1'*∇²f_x0*x1
end

x0 = solution
f_x0 = objective!(model_mg, solution)
∇f_x0 = gradient!(model_mg, solution)
∇²f_x0 = hessian!(model_mg, solution)
∇²f_x0_diag = copy(∇²f_x0)
∇²f_x0_diag[Not(diagind(∇²f_x0_diag))] .= 0


x = collect(-2:.2:2)
y = collect(-1:.2:1)

function vary_parameter(
        model, 
        solution, 
        which_x, 
        which_y, 
        increment_x,
        increment_y)
    varied = copy(solution)
    varied[which_x] += increment_x
    varied[which_y] += increment_y
    sem.objective!(model, varied)
end

function vary_parameter_approx(
        x0, 
        f_x0, 
        ∇f_x0, 
        ∇²f_x0,
        solution,
        which_x, 
        which_y, 
        increment_x,
        increment_y)
    varied = copy(solution)
    varied[which_x] += increment_x
    varied[which_y] += increment_y
    taylor²(x0, f_x0, ∇f_x0, ∇²f_x0, varied)
end

# vary 30 and 31, exact likelihood
plot(x, y, 
    broadcast(
        (x, y) -> vary_parameter(model_mg, solution, 30, 31, x, y),
        x', y)
        )

# 30, 31, approximation
plot(x, y, 
    broadcast(
        (x, y) -> vary_parameter_approx(
            x0, 
            f_x0, 
            ∇f_x0, 
            ∇²f_x0,
            solution, 30, 31, x, y),
        x', y)
        )

# 30, 31, diagonal approximation
plot(x, y, 
    broadcast(
        (x, y) -> vary_parameter_approx(
            x0, 
            f_x0, 
            ∇f_x0, 
            ∇²f_x0_diag,
            solution, 30, 31, x, y),
        x', y)
        )

# vary only 31, all 
plot(x,
    hcat(
        broadcast(
            x -> vary_parameter(
                model_mg, 
                solution, 30, 31, 0.0, x),
            x),
        broadcast(
            x -> vary_parameter_approx(
                x0, 
                f_x0, 
                ∇f_x0, 
                ∇²f_x0,
                solution, 30, 31, 0.0, x),
            x),
        broadcast(
            x -> vary_parameter_approx(
                x0, 
                f_x0, 
                ∇f_x0, 
                ∇²f_x0_diag,
                solution, 30, 31, 0.0, x),
            x)
    ),
    labels = ["exact" "approx" "diag"]
)

# vary only 30, all 
plot(x,
    hcat(
        broadcast(
            x -> vary_parameter(
                model_mg, 
                solution, 30, 31, x, 0.0),
            x),
        broadcast(
            x -> vary_parameter_approx(
                x0, 
                f_x0, 
                ∇f_x0, 
                ∇²f_x0,
                solution, 30, 31, x, 0.0),
            x),
        broadcast(
            x -> vary_parameter_approx(
                x0, 
                f_x0, 
                ∇f_x0, 
                ∇²f_x0_diag,
                solution, 30, 31, x, 0.0),
            x)
    ),
    labels = ["exact" "approx" "diag"]
)

# compare the values
val_approx = broadcast(
    (x, y) -> vary_parameter_approx(
        x0, 
        f_x0, 
        ∇f_x0, 
        ∇²f_x0,
        solution, 30, 31, x, y),
    x', y)

val_diag = broadcast(
    (x, y) -> vary_parameter_approx(
        x0, 
        f_x0, 
        ∇f_x0, 
        ∇²f_x0_diag,
        solution, 30, 31, x, y),
    x', y)

val_approx .> val_diag