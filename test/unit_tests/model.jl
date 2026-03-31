using StructuralEquationModels, Test, Statistics

const SEM = StructuralEquationModels

dat = example_data("political_democracy")
dat_missing = example_data("political_democracy_missing")[:, names(dat)]

obs_vars = [Symbol.("x", 1:3); Symbol.("y", 1:8)]
lat_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin
    # loadings
    ind60 → fixed(1) * x1 + x2 + x3
    dem60 → fixed(1) * y1 + y2 + y3 + y4
    dem65 → fixed(1) * y5 + y6 + y7 + y8
    # latent regressions
    label(:a) * dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60
    # variances
    _(obs_vars) ↔ _(obs_vars)
    _(lat_vars) ↔ _(lat_vars)
    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6
end

ram_matrices =
    RAMMatrices(ParameterTable(graph, observed_vars = obs_vars, latent_vars = lat_vars))

obs = SemObservedData(specification = ram_matrices, data = dat)

function test_vars_api(semobj, spec::SemSpecification)
    @test @inferred(nobserved_vars(semobj)) == nobserved_vars(spec)
    @test observed_vars(semobj) == observed_vars(spec)

    @test @inferred(nlatent_vars(semobj)) == nlatent_vars(spec)
    @test latent_vars(semobj) == latent_vars(spec)

    @test @inferred(nvars(semobj)) == nvars(spec)
    @test vars(semobj) == vars(spec)
end

function test_params_api(semobj, spec::SemSpecification)
    @test @inferred(nparams(semobj)) == nparams(spec)
    @test @inferred(param_labels(semobj)) == param_labels(spec)
end

@testset "Sem(implied=$impliedtype, loss=$losstype)" for (impliedtype, losstype) in [
    (RAM, SemML),
    (RAMSymbolic, SemML),
    (RAMSymbolic, SemWLS),
]
    model = Sem(
        specification = ram_matrices,
        observed = obs,
        implied = impliedtype,
        loss = losstype,
    )

    @test model isa Sem
    @test @inferred(implied(model)) isa impliedtype
    @test @inferred(observed(model)) isa SemObserved

    test_vars_api(model, ram_matrices)
    test_params_api(model, ram_matrices)

    test_vars_api(implied(model), ram_matrices)
    test_params_api(implied(model), ram_matrices)

    @test @inferred(sem_term(model)) isa SemLoss
    @test sem_term(model) isa losstype

    @test @inferred(nsamples(model)) == nsamples(obs)
end

@testset "replace_observed() preserves WLS state through finite-diff wrappers" begin
    model = Sem(
        specification = ram_matrices,
        observed = obs,
        implied = RAMSymbolic,
        loss = SemWLS,
    )
    wls_loss = sem_term(model)
    findiff_model = Sem(SEM.FiniteDiffWrapper(wls_loss))

    new_data = randn(nsamples(obs), nobserved_vars(obs))

    findiff_model_oldstate =
        replace_observed(findiff_model, new_data; recompute_observed_state = false)
    findiff_model_newstate =
        replace_observed(findiff_model, new_data; recompute_observed_state = true)

    loss_orig = SEM._unwrap(sem_term(findiff_model))
    loss_oldstate = SEM._unwrap(sem_term(findiff_model_oldstate))
    loss_newstate = SEM._unwrap(sem_term(findiff_model_newstate))

    @test loss_orig isa SemWLS
    @test loss_oldstate isa SemWLS
    @test loss_newstate isa SemWLS
    @test loss_orig !== loss_oldstate
    @test loss_orig !== loss_newstate
    @test loss_oldstate.V === loss_orig.V
    @test loss_newstate.V !== loss_orig.V
    @test observed_vars(loss_oldstate) == observed_vars(loss_orig)
end

@testset "Sem(...; semterm_column=...) splits ensemble data by group" begin
    dat_grouped = copy(dat[:, [:x1, :x2]])
    n_g1 = size(dat_grouped, 1) ÷ 2
    dat_grouped.group = [fill(:g1, n_g1); fill(:g2, size(dat_grouped, 1) - n_g1)]

    group_graph = @StenoGraph begin
        f1 → fixed(1.0, 1.0) * x1 + label(:λ₂, :λ₂) * x2
        _(Symbol[:x1, :x2]) ↔ _(Symbol[:x1, :x2])
        _(Symbol[:f1]) ↔ _(Symbol[:f1])
    end

    grouped_partable = EnsembleParameterTable(
        group_graph;
        observed_vars = [:x1, :x2],
        latent_vars = [:f1],
        groups = [:g1, :g2],
    )

    grouped_model = Sem(
        specification = grouped_partable,
        data = dat_grouped,
        semterm_column = :group,
        observed = SemObservedData,
        implied = RAM,
        loss = SemML,
    )

    term_g1 = only(filter(term -> SEM.id(term) == :g1, SEM.loss_terms(grouped_model)))
    term_g2 = only(filter(term -> SEM.id(term) == :g2, SEM.loss_terms(grouped_model)))

    @test nsamples(observed(term_g1)) == n_g1
    @test nsamples(observed(term_g2)) == size(dat_grouped, 1) - n_g1
    @test nsamples(grouped_model) == size(dat_grouped, 1)
end
