using StructuralEquationModels, CSV, DataFrames, SparseArrays, Symbolics, LineSearches, Optim, Test, FiniteDiff, LinearAlgebra
import StructuralEquationModels as SEM
include("test_helpers.jl")

############################################################################
### observed data
############################################################################

dat = DataFrame(CSV.File("examples/data/data_dem.csv"))
par_ml = DataFrame(CSV.File("examples/data/par_dem_ml.csv"))
par_ls = DataFrame(CSV.File("examples/data/par_dem_ls.csv"))

############################################################################
### define models
############################################################################

graph_1 = """
    ind60 =∼ 1*x1 + x2 + x3
    dem60 =∼ 1*y1 + y2 + y3 + y4
    dem65 =∼ 1*y5 + y6 + y7 + y8
    dem60 ∼ ind60
    dem65 ∼ dem60
    dem65 ∼ ind60
    ind60 ∼∼ ind60
    dem60 ∼∼ dem60
    dem65 ∼∼ dem65
    x1 ∼∼ x1
    x2 ∼∼ x2
    x3 ∼∼ x3
    y1 ∼∼ y1
    y2 ∼∼ y2
    y3 ∼∼ y3
    y4 ∼∼ y4
    y5 ∼∼ y5
    y6 ∼∼ y6
    y7 ∼∼ y7
    y8 ∼∼ y8
    y1 ∼∼ y5
    y2 ∼∼ y4 + y6
    y3 ∼∼ y7
    y4 ∼∼ y8
    y6 ∼∼ y8
"""

observed_vars = string.([:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8])
latent_vars = ["ind60", "dem60", "dem65"]

partable = ParameterTable(latent_vars, observed_vars, graph_1)

# models
model_ml = Sem(
    specification = partable,
    data = dat
)

model_ls_sym = Sem(
    specification = deepcopy(partable),
    data = dat,
    imply = RAMSymbolic,
    loss = (SemWLS, ),
    start_val = start_simple
)

############################################################################
### test starting values
############################################################################

test_start_val = [fill(0.5, 8); fill(0.05, 3); fill(0.1, 3); fill(1.0, 11); fill(0.05, 6)]
start_val_fabin3 = copy(model_ml.imply.start_val)

update_start!(model_ml)

model_start_partable = Sem(
    specification = partable,
    data = dat,
    start_val = start_parameter_table
)

@testset "start_parameter_table" begin
    @test model_start_partable.imply.start_val ≈ start_val_fabin3
end

############################################################################
### test gradients
############################################################################

@testset "ml_gradients" begin
    @test test_gradient(model_ml, test_start_val)
end

@testset "ls_gradients" begin
    @test test_gradient(model_ls_sym, test_start_val)
end

############################################################################
### test solution
############################################################################

@testset "ml_solution" begin
    solution_ml = sem_fit(model_ml)
    update_partable!(solution_ml)
    @test SEM.compare_estimates(par_ml, partable, 0.01)
    @test SEM.compare_estimates(par_ml, solution_ml.model.specification, 0.01)
end

@testset "ls_solution" begin
    solution_ls = sem_fit(model_ls_sym)
    update_partable!(solution_ls)
    @test !SEM.compare_estimates(par_ml, partable, 0.01)
    @test SEM.compare_estimates(par_ml, solution_ml.model.specification, 0.01)
end

############################################################################
### test sorting
############################################################################

sort!(partable)

model_ml_sorted = Sem(
    specification = partable,
    data = dat
)

@testset "graph sorting" begin
    @test model_ml_sorted.imply.I_A isa LowerTriangular
end

@testset "ml_solution_sorted" begin
    solution_ml_sorted = sem_fit(model_ml_sorted)
    update_partable!(solution_ml_sorted)
    @test SEM.compare_estimates(par_ml, solution_ml_sorted.model.specification, 0.01)
end