using StructuralEquationModels, CSV, DataFrames, Test, StenoGraphs
import StructuralEquationModels as SEM
include("test_helpers.jl")

############################################################################
### observed data
############################################################################

dat = DataFrame(CSV.File("examples/data/data_multigroup.csv"))
par_ml = DataFrame(CSV.File("examples/data/par_multigroup_ml.csv"))
par_ls = DataFrame(CSV.File("examples/data/par_multigroup_ls.csv"))

measures_ml = DataFrame(CSV.File("examples/data/measures_mg_ml.csv"))
measures_ls = DataFrame(CSV.File("examples/data/measures_mg_ls.csv"))

par_ml = filter(row -> (row.free != 0)&(row.op != "~1"), par_ml)
par_ls = filter(row -> (row.free != 0)&(row.op != "~1"), par_ls)

dat_g1 = select(filter(row -> row.school == "Pasteur", dat), Not(:school))
dat_g2 = select(filter(row -> row.school == "Grant-White", dat), Not(:school))

dat = select(dat, Not(:school))

############################################################################
### define models
############################################################################

latent_vars = [:visual, :textual, :speed]
observed_vars = Symbol.(:x, 1:9)

graph = @StenoGraph begin
    # measurement model
    visual  → fixed(1.0, 1.0)*x1 + fixed(0.5,     0.5)*x2 + fixed(0.6, 0.8)*x3
    textual → fixed(1.0, 1.0)*x4 +                     x5 + label(:a₁, :a₂)*x6
    speed   → fixed(1.0, 1.0)*x7 + fixed(1.0,     NaN)*x8 + label(:λ₉, :λ₉)*x9
    # variances and covariances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars)   ↔ _(latent_vars)
    visual ↔ textual + speed
    textual ↔ speed
end

graph = @StenoGraph begin
    # measurement model
    visual  → fixed(1.0, 1.0)*x1 + label(:λ₂, :λ₂)*x2 + label(:λ₃, :λ₃)*x3
    textual → fixed(1.0, 1.0)*x4 + label(:λ₅, :λ₅)*x5 + label(:λ₆, :λ₆)*x6
    speed   → fixed(1.0, 1.0)*x7 + label(:λ₈, :λ₈)*x8 + label(:λ₉, :λ₉)*x9
    # variances and covariances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars)   ⇔ _(latent_vars)
end

partable = EnsembleParameterTable(;
    graph = graph, 
    observed_vars = observed_vars,
    latent_vars = latent_vars,
    groups = [:Pasteur, Symbol("Grant-White")])

ram_matrices = RAMMatrices(partable)

### start values
par_order = [collect(1:6); collect(28:37); [40, 41, 38, 42, 39]; collect(7:16); [19, 20, 17, 21, 18]]

####################################################################
# ML estimation
####################################################################

start_val_ml = Vector{Float64}(par_ml.start[par_order])

model_g1 = Sem(
    specification = ram_matrices[:Pasteur],
    data = dat_g1,
    imply = RAM,
    diff = SemDiffEmpty()
)

model_g2 = Sem(
    specification = ram_matrices[Symbol("Grant-White")],
    data = dat_g2,
    imply = RAMSymbolic,
    diff = SemDiffEmpty()
)

model_ml_multigroup = SemEnsemble(model_g1, model_g2; diff = SemDiffOptim)

############################################################################
### test gradients
############################################################################

using FiniteDiff

@testset "ml_gradients_multigroup" begin
    @test test_gradient(model_ml_multigroup, start_val_ml)
end

# fit
@testset "ml_solution_multigroup" begin
    solution_ml = sem_fit(model_ml_multigroup)
    @test par_ml.est[par_order] ≈ solution_ml.solution rtol = 0.01
end

@testset "fitmeasures/se_ml" begin
    solution_ml = sem_fit(model_ml_multigroup)
    @test all(test_fitmeasures(fit_measures(solution_ml), measures_ml; rtol = 1e-2))
    @test isapprox(par_ml.se[par_order], se_hessian(solution_ml); rtol = 1e-3, atol = 1e-2)
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

using LinearAlgebra

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
    specification = ram_matrices[:Pasteur],
    data = dat_g1,
    imply = RAMSymbolic
)

model_g2 = SemFiniteDiff(
    specification = ram_matrices[Symbol("Grant-White")],
    data = dat_g2,
    imply = RAMSymbolic,
    loss = UserSemML
)

model_ml_multigroup = SemEnsemble(model_g1, model_g2)

@testset "gradients_user_defined_loss" begin
    @test test_gradient(model_ml_multigroup, start_val_ml)
end

# fit
@testset "solution_user_defined_loss" begin
    solution_ml = sem_fit(model_ml_multigroup)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.solution, 0.01)
end

####################################################################
# GLS estimation
####################################################################

model_ls_g1 = Sem(
    specification = ram_matrices[:Pasteur],
    data = dat_g1,
    imply = RAMSymbolic,
    loss = SemWLS
)

model_ls_g2 = Sem(
    specification = ram_matrices[Symbol("Grant-White")],
    data = dat_g2,
    imply = RAMSymbolic,
    loss = SemWLS
)

start_val_ls = Vector{Float64}(par_ls.start[par_order])

model_ls_multigroup = SemEnsemble(model_ls_g1, model_ls_g2)

@testset "ls_gradients_multigroup" begin
    @test test_gradient(model_ls_multigroup, start_val_ls)
end

@testset "ls_solution_multigroup" begin
    solution_ls = sem_fit(model_ls_multigroup)
    @test SEM.compare_estimates(par_ls.est[par_order], solution_ls.solution, 0.01)
end

@testset "fitmeasures/se_ml" begin
    solution_ls = sem_fit(model_ls_multigroup)
    @test all(test_fitmeasures(fit_measures(solution_ls), measures_ls; rtol = 1e-2, fitmeasure_names = fitmeasure_names_ls))
    @test isapprox(par_ls.se[par_order], se_hessian(solution_ls), rtol = 1e-3, atol = 1e-2)
end