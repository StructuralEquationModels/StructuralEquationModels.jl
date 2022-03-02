using StructuralEquationModels, CSV, DataFrames, SparseArrays, Symbolics, LineSearches, Optim, Test, FiniteDiff, LinearAlgebra
import StructuralEquationModels as SEM
include("test_helpers.jl")

############################################################################
### observed data
############################################################################

dat = DataFrame(CSV.File("examples/data/data_dem.csv"))
par_ml = DataFrame(CSV.File("examples/data/par_dem_ml.csv"))
par_ls = DataFrame(CSV.File("examples/data/par_dem_ls.csv"))
measures_ml = DataFrame(CSV.File("examples/data/measures_dem_ml.csv"))
measures_ls = DataFrame(CSV.File("examples/data/measures_dem_ls.csv"))

############################################################################
### define models
############################################################################

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

ram_matrices = RAMMatrices(;
    A = A, 
    S = S, 
    F = F, 
    parameters = x,
    colnames = string.([:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]))

# models
model_ml = Sem(
    specification = ram_matrices,
    data = dat
)

model_ls_sym = Sem(
    specification = ram_matrices,
    data = dat,
    imply = RAMSymbolic,
    loss = (SemWLS, ),
    start_val = start_simple
)

model_ml_sym = Sem(
    specification = ram_matrices,
    data = dat,
    imply = RAMSymbolic
)

model_ridge = Sem(
    specification = ram_matrices,
    data = dat,
    loss = (SemML, SemRidge,),
    α_ridge = .001,
    which_ridge = 16:20
)

model_constant = Sem(
    specification = ram_matrices,
    data = dat,
    loss = (SemML, SemConstant,),
    constant_loss = 3.465
)

############################################################################
### test starting values
############################################################################

par_order = [collect(21:34); collect(15:20); 2;3; 5;6;7; collect(9:14)]
start_val_ml = Vector{Float64}(par_ml.start[par_order])

@test model_ls_sym.imply.start_val == [fill(1.0, 11); fill(0.05, 3); fill(0.0, 6); fill(0.5, 8); fill(0.0, 3)]
@test model_ml.imply.start_val ≈ start_val_ml

############################################################################
### test gradients
############################################################################

@testset "ml_gradients" begin
    @test test_gradient(model_ml, start_val_ml)
end

@testset "ls_gradients" begin
    @test test_gradient(model_ls_sym, start_val_ml)
end

@testset "ridge_gradients" begin
    @test test_gradient(model_ridge, start_val_ml)
end

@testset "constant_gradients" begin
    @test test_gradient(model_constant, start_val_ml)
end

@testset "ml_symbolic_gradients" begin
    @test test_gradient(model_ml_sym, start_val_ml)
end

############################################################################
### test solution
############################################################################

@testset "ml_solution" begin
    solution_ml = sem_fit(model_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.solution, 0.01)
end

@testset "ls_solution" begin
    solution_ls = sem_fit(model_ls_sym)
    @test SEM.compare_estimates(par_ls.est[par_order], solution_ls.solution, 0.01)
end

@testset "constant_solution" begin
    solution_constant = sem_fit(model_constant)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_constant.solution, 0.01)
end

# test constant objective value
@testset "constant_objective_and_gradient" begin
    @test (objective!(model_constant, start_val_ml) - 3.465) ≈ objective!(model_ml, start_val_ml)
    @test gradient!(model_constant, start_val_ml) ≈ gradient!(model_ml, start_val_ml)
end

@testset "ml_symbolic_solution" begin
    solution_ml_symbolic = sem_fit(model_ml_sym)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml_symbolic.solution, 0.01)
end

############################################################################
### test fit assessment
############################################################################

solution_ml = sem_fit(model_ml)

all(test_fitmeasures(fit_measures(solution_ml), measures_ml; rtol = 1e-4))

############################################################################
### test hessians
############################################################################

model_ls = Sem(
    specification = ram_matrices,
    data = dat,
    imply = RAMSymbolic,
    loss = (SemWLS, ),
    hessian = true,
    algorithm = Newton(
        ;linesearch = BackTracking(order=3), 
        alphaguess = InitialHagerZhang())
)

model_ml = Sem(
    specification = ram_matrices,
    data = dat,
    imply = RAMSymbolic,
    hessian = true,
    algorithm = Newton()
)

@testset "ml_hessians" begin
    @test test_hessian(model_ml, start_val_ml)
end

@testset "ls_hessians" begin
    @test test_hessian(model_ls, start_val_ml)
end

@testset "ml_solution_hessian" begin
    solution_ml = sem_fit(model_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.solution, 0.01)
end

@testset "ls_solution_hessian" begin
    solution_ls = sem_fit(model_ls)
    @test SEM.compare_estimates(par_ls.est[par_order], solution_ls.solution, 0.01)
end

############################################################################
### meanstructure
############################################################################

par_ml = DataFrame(CSV.File("examples/data/par_dem_ml_mean.csv"))
par_ls = DataFrame(CSV.File("examples/data/par_dem_ls_mean.csv"))

measures_ml = DataFrame(CSV.File("examples/data/measures_dem_ml_mean.csv"))
measures_ls = DataFrame(CSV.File("examples/data/measures_dem_ls_mean.csv"))

@Symbolics.variables x[1:38]

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

M = [x[32]; x[33]; x[34]; x[35]; x[36]; x[37]; x[38]; x[35]; x[36]; x[37]; x[38]; 0.0; 0.0; 0.0]

ram_matrices = RAMMatrices(;
    A = A, 
    S = S, 
    F = F,
    M = M,
    parameters = x,
    colnames = string.([:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]))

### start values
par_order = [collect(29:42); collect(15:20); 2;3; 5;6;7; collect(9:14); collect(43:45); collect(21:24)]
start_val_ml = Vector{Float64}(par_ml.start[par_order])
start_val_ls = Vector{Float64}(par_ls.start[par_order])

# models
model_ls = Sem(
    specification = ram_matrices,
    data = dat,
    imply = RAMSymbolic,
    loss = (SemWLS, ),
    meanstructure = true,
    start_val = start_val_ls
)

model_ml = Sem(
    specification = ram_matrices,
    data = dat,
    meanstructure = true,
    start_val = start_val_ml
)

model_ml_sym = Sem(
    specification = ram_matrices,
    data = dat,
    imply = RAMSymbolic,
    meanstructure = true,
    start_val = start_val_ml
)

############################################################################
### test solution
############################################################################

@testset "ml_solution_meanstructure" begin
    solution_ml = sem_fit(model_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.solution, 0.01)
end

@testset "ls_solution_meanstructure" begin
    solution_ls = sem_fit(model_ls)
    @test SEM.compare_estimates(par_ls.est[par_order], solution_ls.solution, 0.01)
end

@testset "ml_solution_meanstructure_nsymbolic" begin
    solution_ml_symbolic = sem_fit(model_ml_sym)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml_symbolic.solution, 0.01)
end

############################################################################
### test gradients
############################################################################

@testset "ml_gradients_meanstructure" begin
    @test test_gradient(model_ml, start_val_ml)
end

@testset "ls_gradients_meanstructure" begin
    @test test_gradient(model_ls, start_val_ls)
end

@testset "ml_gradients_meanstructure_symbolic" begin
    @test test_gradient(model_ml_sym, start_val_ml)
end

############################################################################
### test fit assessment
############################################################################

solution_ml = sem_fit(model_ml)

# F value
F = Fₘᵢₙ(solution_ml)
F_lav = measures_ml.x[measures_ml.Column1 .==  "fmin"][1]
isapprox(F, 2F_lav; rtol = 1e-4)

# χ²
chi2 = χ²(solution_ml)
chi2_lav = measures_ml.x[measures_ml.Column1 .==  "chisq"][1]
isapprox(chi2, chi2_lav; rtol = 1e-4)

# df
df_ = df(solution_ml)
df_lav = measures_ml.x[measures_ml.Column1 .==  "df"][1]
isapprox(df_, df_lav; rtol = 1e-10)

# p
p_val = p_value(solution_ml)
p_val_lav = measures_ml.x[measures_ml.Column1 .==  "pvalue"][1]
isapprox(p_val, p_val_lav; rtol = 1e-4)

# se
se = se_fisher(solution_ml)
maximum(abs.(par_ml.se[par_order] .- se))
isapprox(par_ml.se[par_order], se; rtol = 1e-3)

# se_fisher
###
###

# -2ll
m2ll = minus2ll(solution_ml)
logl_lav = measures_ml.x[measures_ml.Column1 .==  "logl"][1]
isapprox(m2ll, -2logl_lav; rtol = 1e-4)

# AIC
aic = AIC(solution_ml)
aic_lav = measures_ml.x[measures_ml.Column1 .==  "aic"][1]
isapprox(aic, aic_lav; rtol = 1e-4)

# BIC
bic = BIC(solution_ml)
bic_lav = measures_ml.x[measures_ml.Column1 .==  "bic"][1]
isapprox(bic, bic_lav; rtol = 1e-4)

# RMSEA
rmsea = RMSEA(solution_ml)
rmsea_lav = measures_ml.x[measures_ml.Column1 .==  "rmsea"][1]
isapprox(rmsea, rmsea_lav; rtol = 1e-4)

fm = fit_measures(solution_ml)




############################################################################
### fiml
############################################################################

############################################################################
### observed data
############################################################################

dat = DataFrame(CSV.read("examples/data/data_dem_fiml.csv", DataFrame; missingstring = "NA"))
par_ml = DataFrame(CSV.read("examples/data/par_dem_ml_fiml.csv", DataFrame))

############################################################################
### define models
############################################################################

@Symbolics.variables x[1:38]

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

M = [x[32]; x[33]; x[34]; x[35]; x[36]; x[37]; x[38]; x[35]; x[36]; x[37]; x[38]; 0.0; 0.0; 0.0]

ram_matrices = RAMMatrices(;
    A = A, 
    S = S, 
    F = F,
    M = M,
    parameters = x,
    colnames = string.([:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]))

### start values
par_order = [collect(29:42); collect(15:20); 2;3; 5;6;7; collect(9:14); collect(43:45); collect(21:24)]
start_val_ml = Vector{Float64}(par_ml.start[par_order])

# models
model_ml = Sem(
    specification = ram_matrices,
    data = dat,
    observed = SemObsMissing,
    loss = (SemFIML,),
    start_val = start_val_ml
)

model_ml_sym = Sem(
    specification = ram_matrices,
    data = dat,
    observed = SemObsMissing,
    imply = RAMSymbolic,
    loss = (SemFIML,),
    start_val = start_val_ml
)

############################################################################
### test gradients
############################################################################

@testset "fiml_gradient" begin
    @test test_gradient(model_ml, start_val_ml)
end

@testset "fiml_gradient_symbolic" begin
    @test test_gradient(model_ml_sym, start_val_ml)
end

############################################################################
### test solution
############################################################################

@testset "fiml_solution" begin
    solution_ml = sem_fit(model_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.solution, 0.01)
end

@testset "fiml_solution_symbolic" begin
    solution_ml_symbolic = sem_fit(model_ml_sym)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml_symbolic.solution, 0.01)
end