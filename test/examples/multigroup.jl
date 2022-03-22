using StructuralEquationModels, CSV, DataFrames, SparseArrays, Symbolics, LineSearches, Optim, Test, LinearAlgebra
import StructuralEquationModels as SEM
include("test_helpers.jl")

############################################################################
### observed data
############################################################################

dat = DataFrame(CSV.File("examples/data/data_multigroup.csv"))
par_ml = DataFrame(CSV.File("examples/data/par_multigroup_ml.csv"))
par_ls = DataFrame(CSV.File("examples/data/par_multigroup_ls.csv"))

par_ml = filter(row -> (row.free != 0)&(row.op != "~1"), par_ml)
par_ls = filter(row -> (row.free != 0)&(row.op != "~1"), par_ls)

dat_g1 = select(filter(row -> row.school == "Pasteur", dat), Not(:school))
dat_g2 = select(filter(row -> row.school == "Grant-White", dat), Not(:school))

dat = select(dat, Not(:school))

############################################################################
### define models
############################################################################

x = Symbol.(:x, 1:36)
#x = [x...]

F = zeros(9, 12)
F[diagind(F)] .= 1.0

A = Matrix{Any}(zeros(12, 12))
A[1, 10] = 1.0; A[4, 11] = 1.0; A[7, 12] = 1.0
A[2:3, 10] .= [x...][16:17]; A[5:6, 11] .= [x...][18:19]; A[8:9, 12] .= [x...][20:21]; 

# group 1
S1 = Matrix{Any}(zeros(12, 12))
S1[diagind(S1)] .= [x...][1:12]
S1[10, 11] = x[13]; S1[11, 10] = x[13]
S1[10, 12] = x[14]; S1[12, 10] = x[14]
S1[12, 11] = x[15]; S1[11, 12] = x[15]

# group 2
S2 = Matrix{Any}(zeros(12, 12))
S2[diagind(S2)] .= [x...][22:33]
S2[10, 11] = x[34]; S2[11, 10] = x[34]
S2[10, 12] = x[35]; S2[12, 10] = x[35]
S2[12, 11] = x[36]; S2[11, 12] = x[36]

ram_matrices_g1 = RAMMatrices(;
    A = A,
    S = S1,
    F = F,
    parameters = x,
    colnames = string.([:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :visual, :textual, :speed]))

ram_matrices_g2 = RAMMatrices(;
    A = A,
    S = S2,
    F = F,
    parameters = x,
    colnames = string.([:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :visual, :textual, :speed]))

### start values
par_order = [collect(7:21); collect(1:6); collect(28:42)]

####################################################################
# ML estimation
####################################################################

start_val_ml = Vector{Float64}(par_ml.start[par_order])

model_g1 = Sem(
    specification = ram_matrices_g1,
    data = dat_g1,
    imply = RAMSymbolic
)

model_g2 = Sem(
    specification = ram_matrices_g2,
    data = dat_g2,
    imply = RAMSymbolic
)

model_ml_multigroup = SemEnsemble((model_g1, model_g2), SemDiffOptim(), start_val_ml)

############################################################################
### test gradients
############################################################################

using FiniteDiff

@testset "ml_gradients_multigroup" begin
    @test test_gradient(model_ml_multigroup, start_val_ml)
end

# fit
@testset "ml_solution_multigroup" begin
    solution_ml = sem_fit(model_ml_multigroup; start_val = start_val_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.solution, 0.01)
end

####################################################################
# ML estimation - without Gradients and Hessian
####################################################################

struct UserSemML <: SemLossFunction
    F
    G
    H
end 

############################################################################
### constructor
############################################################################

UserSemML(;n_par, kwargs...) = UserSemML([1.0], zeros(n_par), zeros(n_par, n_par)) 

############################################################################
### functors
############################################################################

function (semml::UserSemML)(par, F, G, H, model)
    if G error("analytic gradient of ML is not implemented (yet)") end
    if H error("analytic hessian of ML is not implemented (yet)") end

    a = cholesky(Symmetric(model.imply.Σ); check = false)
    if !isposdef(a)
        semml.F[1] = Inf
    else
        ld = logdet(a)
        Σ_inv = LinearAlgebra.inv(a)
        if !isnothing(F)
            prod = Σ_inv*model.observed.obs_cov
            semml.F[1] = ld + tr(prod)
        end
    end
end

start_val_ml = Vector{Float64}(par_ml.start[par_order])

# models
model_g1 = Sem(
    specification = ram_matrices_g1,
    data = dat_g1,
    imply = RAMSymbolic
)

model_g2 = SemFiniteDiff(
    specification = ram_matrices_g2,
    data = dat_g2,
    imply = RAMSymbolic,
    loss = UserSemML
)

model_ml_multigroup = SemEnsemble((model_g1, model_g2), SemDiffOptim(), start_val_ml)

@testset "gradients_user_defined_loss" begin
    @test test_gradient(model_ml_multigroup, start_val_ml)
end

# fit
@testset "solution_user_defined_loss" begin
    solution_ml = sem_fit(model_ml_multigroup; start_val = start_val_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.solution, 0.01)
end

####################################################################
# GLS estimation
####################################################################

model_ls_g1 = Sem(
    specification = ram_matrices_g1,
    data = dat_g1,
    imply = RAMSymbolic,
    loss = SemWLS
)

model_ls_g2 = Sem(
    specification = ram_matrices_g2,
    data = dat_g2,
    imply = RAMSymbolic,
    loss = SemWLS
)

start_val_ls = Vector{Float64}(par_ls.start[par_order])

model_ls_multigroup = SemEnsemble((model_ls_g1, model_ls_g2), SemDiffOptim(), start_val_ls)

@testset "ls_gradients_multigroup" begin
    @test test_gradient(model_ls_multigroup, start_val_ls)
end

@testset "ls_solution_multigroup" begin
    solution_ls = sem_fit(model_ls_multigroup; start_val = start_val_ls)
    @test SEM.compare_estimates(par_ls.est[par_order], solution_ls.solution, 0.01)
end