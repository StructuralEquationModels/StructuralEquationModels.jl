using StructuralEquationModels, Test, FiniteDiff
# import StructuralEquationModels as SEM
include("helper.jl")

############################################################################
### data
############################################################################

dat = example_data("political_democracy")
dat_missing = example_data("political_democracy_missing")
solution_lav = example_data("political_democracy_solution")

############################################################################
### specification
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

A =[0  0  0  0  0  0  0  0  0  0  0     1.0   0     0
    0  0  0  0  0  0  0  0  0  0  0     :x21  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x22  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1.0   0
    0  0  0  0  0  0  0  0  0  0  0     0     :x23  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x24  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x25  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x26
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x27
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x28
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     :x29  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x30  :x31  0]

spec = RAMMatrices(;
    A = A, 
    S = S, 
    F = F, 
    parameters = x,
    colnames = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8, :ind60, :dem60, :dem65]
)

partable = ParameterTable(spec)

model_ml = Sem(
    specification = spec,
    data = dat
)

model_ls_sym = Sem(
    specification = spec,
    data = dat,
    imply = RAMSymbolic,
    loss = SemWLS,
    start_val = start_simple
)

model_ml_sym = Sem(
    specification = spec,
    data = dat,
    imply = RAMSymbolic
)

model_ridge = Sem(
    specification = spec,
    data = dat,
    loss = (SemML, SemRidge),
    α_ridge = .001,
    which_ridge = 16:20
)

model_ridge_id = Sem(
    specification = spec,
    data = dat,
    loss = (SemML, SemRidge),
    α_ridge = .001,
    which_ridge = [:x16, :x17, :x18, :x19, :x20]
)

model_constant = Sem(
    specification = spec,
    data = dat,
    loss = (SemML, SemConstant),
    constant_loss = 3.465
)

############################################################################
### test gradients
############################################################################

start_test = [fill(1.0, 11); fill(0.05, 3); fill(0.05, 6); fill(0.5, 8); fill(0.05, 3)]

models = [model_ml, model_ls_sym, model_ridge, model_ridge_id, model_constant, model_ml_sym]
names = ["ml", "ls_sym", "ridge", "ridge_id", "constant", "ml_sym"]

for (model, name) in zip(models, names)
    try
        @testset "$(name)_gradient" begin
            @test test_gradient(model, start_test; rtol = 1e-9)
        end
    catch
    end
end

############################################################################
### test solution
############################################################################

models = [model_ml, model_ls_sym, model_ml_sym, model_constant]
names = ["ml", "ls_sym", "ml_sym", "constant"]
solution_names = Symbol.("parameter_estimates_".*["ml", "ls", "ml", "ml"])

for (model, name, solution_name) in zip(models, names, solution_names)
    try
        @testset "$(name)_solution" begin
            solution = sem_fit(model)
            update_estimate!(partable, solution)
            @test compare_estimates(partable, solution_lav[solution_name]; atol = 1e-2)
        end
    catch
    end
end

@testset "ridge_solution_id" begin
    solution_ridge = sem_fit(model_ridge)
    solution_ridge_id = sem_fit(model_ridge_id)
    @test solution_ridge.solution ≈ solution_ridge_id.solution atol = 1e-6
end

# test constant objective value
@testset "constant_objective_and_gradient" begin
    @test (objective!(model_constant, start_test) - 3.465) ≈ objective!(model_ml, start_test)
    @test gradient!(model_constant, start_test) ≈ gradient!(model_ml, start_test)
end

############################################################################
### test fit assessment
############################################################################

@testset "fitmeasures/se_ml" begin
    solution_ml = sem_fit(model_ml)
    @test all(test_fitmeasures(fit_measures(solution_ml), solution_lav[:fitmeasures_ml]; atol = 1e-3))

    update_partable!(partable, identifier(model_ml), se_hessian(solution_ml), :se)
    @test compare_estimates(partable, solution_lav[:parameter_estimates_ml]; atol = 1e-3, col = :se, lav_col = :se)
end

@testset "fitmeasures/se_ls" begin
    solution_ls = sem_fit(model_ls_sym)
    fm = fit_measures(solution_ls)
    @test all(test_fitmeasures(fm, solution_lav[:fitmeasures_ls]; atol = 1e-3, fitmeasure_names = fitmeasure_names_ls))
    @test (fm[:AIC] === missing) & (fm[:BIC] === missing) & (fm[:minus2ll] === missing)

    update_partable!(partable, identifier(model_ls_sym), se_hessian(solution_ls), :se)
    @test compare_estimates(partable, solution_lav[:parameter_estimates_ls]; atol = 1e-2, col = :se, lav_col = :se)
end

############################################################################
### test hessians
############################################################################

using Optim, LineSearches

model_ls = Sem(
    specification = spec,
    data = dat,
    imply = RAMSymbolic,
    loss = SemWLS,
    hessian = true,
    algorithm = Newton(
        ;linesearch = BackTracking(order=3), 
        alphaguess = InitialHagerZhang())
)

model_ml = Sem(
    specification = spec,
    data = dat,
    imply = RAMSymbolic,
    hessian = true,
    algorithm = Newton()
)

@testset "ml_hessians" begin
    @test test_hessian(model_ml, start_test; atol = 1e-4)
end

@testset "ls_hessians" begin
    @test test_hessian(model_ls, start_test; atol = 1e-4)
end

@testset "ml_solution_hessian" begin
    solution = sem_fit(model_ml)
    update_estimate!(partable, solution)
    @test compare_estimates(partable, solution_lav[:parameter_estimates_ml]; atol = 1e-3)
end

@testset "ls_solution_hessian" begin
    solution = sem_fit(model_ls)
    update_estimate!(partable, solution)
    @test compare_estimates(partable, solution_lav[:parameter_estimates_ls]; atol = 1e-3)
end

############################################################################
### meanstructure
############################################################################

x = Symbol.("x".*string.(1:38))

M = [:x32; :x33; :x34; :x35; :x36; :x37; :x38; :x35; :x36; :x37; :x38; 0.0; 0.0; 0.0]

spec_mean = RAMMatrices(;
    A = A, 
    S = S, 
    F = F,
    M = M,
    parameters = x,
    colnames = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8, :ind60, :dem60, :dem65])

partable = ParameterTable(spec_mean)

# starting values
start_test = [fill(1.0, 11); fill(0.05, 3); fill(0.05, 6); fill(0.5, 8); fill(0.05, 3); fill(0.1, 7)]

# models
model_ls = Sem(
    specification = spec_mean,
    data = dat,
    imply = RAMSymbolic,
    loss = SemWLS,
    meanstructure = true
)

model_ml = Sem(
    specification = spec_mean,
    data = dat,
    meanstructure = true
)

model_ml_sym = Sem(
    specification = spec_mean,
    data = dat,
    imply = RAMSymbolic,
    meanstructure = true,
    start_val = start_test
)

############################################################################
### test gradients
############################################################################

models = [model_ml, model_ls, model_ml_sym]
names = ["ml", "ls_sym", "ml_sym"]

for (model, name) in zip(models, names)
    try
        @testset "$(name)_gradient_mean" begin
            @test test_gradient(model, start_test; rtol = 1e-9)
        end
    catch
    end
end

############################################################################
### test solution
############################################################################

solution_names = Symbol.("parameter_estimates_".*["ml", "ls", "ml"].*"_mean")

for (model, name, solution_name) in zip(models, names, solution_names)
    try
        @testset "$(name)_solution_mean" begin
            solution = sem_fit(model)
            update_estimate!(partable, solution)
            @test compare_estimates(partable, solution_lav[solution_name]; atol = 1e-2)
        end
    catch
    end
end

############################################################################
### test fit assessment
############################################################################

@testset "fitmeasures/se_ml_mean" begin
    solution_ml = sem_fit(model_ml)
    @test all(test_fitmeasures(fit_measures(solution_ml), solution_lav[:fitmeasures_ml_mean]; atol = 1e-3))

    update_partable!(partable, identifier(model_ml), se_hessian(solution_ml), :se)
    @test compare_estimates(partable, solution_lav[:parameter_estimates_ml_mean]; atol = 1e-3, col = :se, lav_col = :se)
end

@testset "fitmeasures/se_ls_mean" begin
    solution_ls = sem_fit(model_ls)
    fm = fit_measures(solution_ls)
    @test all(test_fitmeasures(fm, solution_lav[:fitmeasures_ls_mean]; atol = 1e-3, fitmeasure_names = fitmeasure_names_ls))
    @test (fm[:AIC] === missing) & (fm[:BIC] === missing) & (fm[:minus2ll] === missing)

    update_partable!(partable, identifier(model_ls), se_hessian(solution_ls), :se)
    @test compare_estimates(partable, solution_lav[:parameter_estimates_ls_mean]; atol = 1e-2, col = :se, lav_col = :se)
end

############################################################################
### fiml
############################################################################

# models
model_ml = Sem(
    specification = spec_mean,
    data = dat_missing,
    observed = SemObsMissing,
    loss = SemFIML
)

model_ml_sym = Sem(
    specification = spec_mean,
    data = dat_missing,
    observed = SemObsMissing,
    imply = RAMSymbolic,
    loss = SemFIML,
    start_val = start_test
)

############################################################################
### test gradients
############################################################################

@testset "fiml_gradient" begin
    @test test_gradient(model_ml, start_test; atol = 1e-6)
end

@testset "fiml_gradient_symbolic" begin
    @test test_gradient(model_ml_sym, start_test; atol = 1e-6)
end

############################################################################
### test solution
############################################################################

@testset "fiml_solution" begin
    solution = sem_fit(model_ml)
    update_estimate!(partable, solution)
    @test compare_estimates(partable, solution_lav[:parameter_estimates_fiml]; atol = 1e-2)
end

@testset "fiml_solution_symbolic" begin
    solution = sem_fit(model_ml_sym)
    update_estimate!(partable, solution)
    @test compare_estimates(partable, solution_lav[:parameter_estimates_fiml]; atol = 1e-2)
end

############################################################################
### test fit measures
############################################################################

@testset "fitmeasures/se_fiml" begin
    solution_ml = sem_fit(model_ml)
    @test all(test_fitmeasures(fit_measures(solution_ml), solution_lav[:fitmeasures_fiml]; atol = 1e-3))

    update_partable!(partable, identifier(model_ml), se_hessian(solution_ml), :se)
    @test compare_estimates(partable, solution_lav[:parameter_estimates_fiml]; atol = 1e-3, col = :se, lav_col = :se)
end