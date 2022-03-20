using StructuralEquationModels, CSV, DataFrames, Test, FiniteDiff
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

x = Symbol.("x".*string.(1:31))

S =[:x1   0    0     0     0      0     0     0     0     0     0     0     0     0
    0     :x2  0     0     0      0     0     0     0     0     0     0     0     0
    0     0     :x3  0     0      0     0     0     0     0     0     0     0     0
    0     0     0     :x4  0      0     0     :x15  0     0     0     0     0     0
    0     0     0     0     :x5   0     :x16  0     :x17  0     0     0     0     0
    0     0     0     0     0     :x6  0      0     0     :x18  0     0     0     0
    0     0     0     0     :x16  0     :x7   0     0     0     :x19  0     0     0
    0     0     0     :x15 0      0     0     :x8   0     0     0     0     0     0
    0     0     0     0     :x17  0     0     0     :x9   0     :x20  0     0     0
    0     0     0     0     0     :x18 0      0     0     :x10  0     0     0     0
    0     0     0     0     0     0     :x19  0     :x20  0     :x11  0     0     0
    0     0     0     0     0     0     0     0     0     0     0     :x12  0     0
    0     0     0     0     0     0     0     0     0     0     0     0     :x13  0
    0     0     0     0     0     0     0     0     0     0     0     0     0     :x14]

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

A =[0  0  0  0  0  0  0  0  0  0  0     1.0     0     0
    0  0  0  0  0  0  0  0  0  0  0     :x21  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x22  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1.0     0
    0  0  0  0  0  0  0  0  0  0  0     0     :x23  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x24  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x25  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1.0
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x26
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x27
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x28
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     :x29  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x30  :x31  0]

ram_matrices = RAMMatrices(;
    A = A, 
    S = S, 
    F = F, 
    parameters = x,
    colnames = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8, :ind60, :dem60, :dem65])

# models
model_ml = Sem(
    specification = ram_matrices,
    data = dat,
    diff = SemDiffNLopt
)

model_ls_sym = Sem(
    specification = ram_matrices,
    data = dat,
    imply = RAMSymbolic,
    loss = (SemWLS, ),
    diff = SemDiffNLopt
)

model_ridge = Sem(
    specification = ram_matrices,
    data = dat,
    loss = (SemML, SemRidge,),
    diff = SemDiffNLopt,
    α_ridge = .001,
    which_ridge = [:x16, :x17, :x18, :x19, :x20]
)

# NLopt constraints -----------------------------------------------

# 1.5*x1 == x2 (aka 1.5*x1 - x2 == 0)
#= function eq_constraint(x, grad)
    if length(grad) > 0
        grad .= 0.0
        grad[1] = 1.5
        grad[2] = -1.0
    end
    1.5*x[1] - x[2]
end =#

# x30 ≥ 1.0 (aka 1.0 - x30 ≤ 0)
#= function ineq_constraint(x, grad)
    if length(grad) > 0
        grad .= 0.0
        grad[30] = -1.0
    end
    1.0 - x[30]
end =#

# x30*x31 ≥ 0.6 (aka 0.6 - x30*x31 ≤ 0)
function ineq_constraint(x, grad)
    if length(grad) > 0
        grad .= 0.0
        grad[30] = -x[31]
        grad[31] = -x[30]
    end
    0.6 - x[30]*x[31]
end

constrained_diff = SemDiffNLopt(;
    algorithm = :AUGLAG,
    local_algorithm = :LD_LBFGS,
    local_options = Dict(
        :ftol_rel => 1e-6
    ),
    # equality_constraints = NLoptConstraint(;f = eq_constraint, tol = 1e-14),
    inequality_constraints = NLoptConstraint(;f = ineq_constraint, tol = 0.0),
)

model_ml_constrained = Sem(
    specification = ram_matrices,
    data = dat,
    diff = constrained_diff
)

solution_constrained = sem_fit(model_ml_constrained)

#solution_ml = sem_fit(model_ml)


# NLopt option setting --------------------------------------------

model_ml_maxeval = Sem(
    specification = ram_matrices,
    data = dat,
    diff = SemDiffNLopt,
    options = Dict(:maxeval => 10)
)

############################################################################
### test starting values
############################################################################

par_order = [collect(21:34); collect(15:20); 2;3; 5;6;7; collect(9:14)]
start_val_ml = Vector{Float64}(par_ml.start[par_order])

@test start_simple(model_ls_sym) == [fill(1.0, 11); fill(0.05, 3); fill(0.0, 6); fill(0.5, 8); fill(0.0, 3)]
@test start_val(model_ml) ≈ start_val_ml

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

############################################################################
### test solution
############################################################################

@testset "ml_solution" begin
    solution_ml = sem_fit(model_ml)
    @test isapprox(par_ml.est[par_order], solution_ml.solution; rtol = 0.01)
end

@testset "ls_solution" begin
    solution_ls = sem_fit(model_ls_sym)
    @test SEM.compare_estimates(par_ls.est[par_order], solution_ls.solution, 0.01)
end

@testset "ml_solution_maxeval" begin
    solution_maxeval = sem_fit(model_ml_maxeval)
    @test solution_maxeval.optimization_result.problem.numevals == 10
    @test solution_maxeval.optimization_result.result[3] == :MAXEVAL_REACHED
end

@testset "ml_solution_constrained" begin
    solution_constrained = sem_fit(model_ml_constrained)
    @test solution_constrained.solution[31]*solution_constrained.solution[30] >= 0.6
    @test all(abs.(solution_constrained.solution) .< 10)
    @test_skip solution_constrained.optimization_result.result[3] == :FTOL_REACHED
    @test abs(solution_constrained.minimum - 21.21) < 0.01
end

############################################################################
### test fit assessment
############################################################################

@testset "fitmeasures/se_ml" begin
    solution_ml = sem_fit(model_ml)
    @test all(test_fitmeasures(fit_measures(solution_ml), measures_ml; rtol = 1e-2))
    @test par_ml.se[par_order] ≈ se_hessian(solution_ml) rtol = 1e-3
end

@testset "fitmeasures/se_ls" begin
    solution_ls = sem_fit(model_ls_sym)
    @test all(test_fitmeasures(fit_measures(solution_ls), measures_ls; rtol = 1e-2, fitmeasure_names = fitmeasure_names_ls))
    @test_skip par_ls.se[par_order] ≈ se_hessian(solution_ls) rtol = 1e-3
end