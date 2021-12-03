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
lossfun_c = SemConstant(-(logdet(semobserved.obs_cov) + 18), length(start_val))

loss_ridge = SemLoss(
    (
        lossfun_ml,
        lossfun_c,
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
lossfun_c = SemConstant(-(logdet(semobserved.obs_cov) + 18), length(start_val))

loss_ridge = SemLoss(
    (
        lossfun_ml,
        lossfun_c,
        SemRidge(.02, [1, 2, 6, 7, 8], length(start_val))
    )
)

model_ridge = Sem(semobserved, imply, loss_ridge, semdiff)

solution = sem_fit(model_ridge)

using ProfileView

