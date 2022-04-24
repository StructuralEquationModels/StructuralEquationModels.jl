####################################################################
# ML estimation
####################################################################

model_g1 = Sem(
    specification = specification_g1,
    data = dat_g1,
    imply = RAMSymbolic
)

model_g2 = Sem(
    specification = specification_g2,
    data = dat_g2,
    imply = RAM
)

model_ml_multigroup = SemEnsemble(model_g1, model_g2; diff = semdiff)

############################################################################
### test gradients
############################################################################

@testset "ml_gradients_multigroup" begin
    @test test_gradient(model_ml_multigroup, start_test; atol = 1e-9)
end

# fit
@testset "ml_solution_multigroup" begin
    solution = sem_fit(model_ml_multigroup)
    update_estimate!(partable, solution)
    @test compare_estimates(
        partable, 
        solution_lav[:parameter_estimates_ml]; atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2))
end

@testset "fitmeasures/se_ml" begin
    solution_ml = sem_fit(model_ml_multigroup)
    @test all(test_fitmeasures(
        fit_measures(solution_ml), 
        solution_lav[:fitmeasures_ml]; rtol = 1e-2, atol = 1e-7))

    update_partable!(
        partable, identifier(model_ml_multigroup), se_hessian(solution_ml), :se)
    @test compare_estimates(
        partable, 
        solution_lav[:parameter_estimates_ml]; atol = 1e-3, 
        col = :se, lav_col = :se,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2))
end

####################################################################
# ML estimation - user defined loss function
####################################################################

struct UserSemML <: SemLossFunction
    objective
    gradient
    hessian
end

############################################################################
### constructor
############################################################################

UserSemML(;n_par, kwargs...) = UserSemML([1.0], zeros(n_par), zeros(n_par, n_par)) 

############################################################################
### functors
############################################################################

import LinearAlgebra: Symmetric, cholesky, isposdef, logdet, tr
import LinearAlgebra

function (semml::UserSemML)(par, F, G, H, model)
    if G error("analytic gradient of ML is not implemented (yet)") end
    if H error("analytic hessian of ML is not implemented (yet)") end

    a = cholesky(Symmetric(model.imply.Σ); check = false)
    if !isposdef(a)
        semml.objective[1] = Inf
    else
        ld = logdet(a)
        Σ_inv = LinearAlgebra.inv(a)
        if !isnothing(F)
            prod = Σ_inv*model.observed.obs_cov
            semml.objective[1] = ld + tr(prod)
        end
    end
end

# models
model_g1 = Sem(
    specification = specification_g1,
    data = dat_g1,
    imply = RAMSymbolic
)

model_g2 = SemFiniteDiff(
    specification = specification_g2,
    data = dat_g2,
    imply = RAMSymbolic,
    loss = UserSemML
)

model_ml_multigroup = SemEnsemble(model_g1, model_g2; diff = semdiff)

@testset "gradients_user_defined_loss" begin
    @test test_gradient(model_ml_multigroup, start_test; atol = 1e-9)
end

# fit
@testset "solution_user_defined_loss" begin
    solution = sem_fit(model_ml_multigroup)
    update_estimate!(partable, solution)
    @test compare_estimates(
        partable, 
        solution_lav[:parameter_estimates_ml]; atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2))
end

####################################################################
# GLS estimation
####################################################################

model_ls_g1 = Sem(
    specification = specification_g1,
    data = dat_g1,
    imply = RAMSymbolic,
    loss = SemWLS
)

model_ls_g2 = Sem(
    specification = specification_g2,
    data = dat_g2,
    imply = RAMSymbolic,
    loss = SemWLS
)

model_ls_multigroup = SemEnsemble(model_ls_g1, model_ls_g2; diff = semdiff)

@testset "ls_gradients_multigroup" begin
    @test test_gradient(model_ls_multigroup, start_test; atol = 1e-9)
end

@testset "ls_solution_multigroup" begin
    solution = sem_fit(model_ls_multigroup)
    update_estimate!(partable, solution)
    @test compare_estimates(
        partable, 
        solution_lav[:parameter_estimates_ls]; atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2))
end

@testset "fitmeasures/se_ls" begin
    solution_ls = sem_fit(model_ls_multigroup)
    @test all(test_fitmeasures(
        fit_measures(solution_ls), 
        solution_lav[:fitmeasures_ls];
        fitmeasure_names = fitmeasure_names_ls, rtol = 1e-2, atol = 1e-5))

    update_partable!(
        partable, identifier(model_ls_multigroup), se_hessian(solution_ls), :se)
    @test compare_estimates(
        partable, 
        solution_lav[:parameter_estimates_ls]; atol = 1e-2,
        col = :se, lav_col = :se,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2))
end

if !isnothing(specification_miss_g1)

####################################################################
# FIML estimation
####################################################################

model_g1 = Sem(
    specification = specification_miss_g1,
    observed = SemObsMissing,
    loss = SemFIML,
    data = dat_miss_g1,
    imply = RAM,
    diff = SemDiffEmpty()
)

model_g2 = Sem(
    specification = specification_miss_g2,
    observed = SemObsMissing,
    loss = SemFIML,
    data = dat_miss_g2,
    imply = RAM,
    diff = SemDiffEmpty()
)

model_ml_multigroup = SemEnsemble(model_g1, model_g2; diff = semdiff)

############################################################################
### test gradients
############################################################################

start_test = [
    fill(0.5, 6); 
    fill(1.0, 9); 
    0.05; 0.01; 0.01; 0.05; 0.01; 0.05; 
    fill(0.01, 9);
    fill(1.0, 9); 
    0.05; 0.01; 0.01; 0.05; 0.01; 0.05;
    fill(0.01, 9)]

@testset "fiml_gradients_multigroup" begin
    @test test_gradient(model_ml_multigroup, start_test; atol = 1e-7)
end


@testset "fiml_solution_multigroup" begin
    solution = sem_fit(model_ml_multigroup)
    update_estimate!(partable_miss, solution)
    @test compare_estimates(
        partable_miss, 
        solution_lav[:parameter_estimates_fiml]; atol = 1e-4,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2))
end

@testset "fitmeasures/se_fiml" begin
    solution = sem_fit(model_ml_multigroup)
    @test all(test_fitmeasures(
        fit_measures(solution), 
        solution_lav[:fitmeasures_fiml]; rtol = 1e-3, atol = 0))

    update_partable!(
        partable_miss, identifier(model_ml_multigroup), se_hessian(solution), :se)
    @test compare_estimates(
        partable_miss, 
        solution_lav[:parameter_estimates_fiml]; atol = 1e-3,
        col = :se, lav_col = :se,
        lav_groups = Dict(:Pasteur => 1, :Grant_White => 2))
end

end