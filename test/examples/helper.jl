using LinearAlgebra: norm
using Suppressor

function is_extended_tests()
    return lowercase(get(ENV, "JULIA_EXTENDED_TESTS", "false")) == "true"
end

function test_gradient(model, params; rtol = 1e-10, atol = 0)
    @test nparams(model) == length(params)

    true_grad = FiniteDiff.finite_difference_gradient(Base.Fix1(objective!, model), params)
    gradient = similar(params)

    # F and G
    fill!(gradient, NaN)
    gradient!(gradient, model, params)
    @test gradient ≈ true_grad rtol = rtol atol = atol

    # only G
    fill!(gradient, NaN)
    objective_gradient!(gradient, model, params)
    @test gradient ≈ true_grad rtol = rtol atol = atol
end

function test_hessian(model, params; rtol = 1e-4, atol = 0)
    true_hessian =
        FiniteDiff.finite_difference_hessian(Base.Fix1(objective!, model), params)
    hessian = similar(params, size(true_hessian))
    gradient = similar(params)

    # H
    fill!(hessian, NaN)
    hessian!(hessian, model, params)
    @test hessian ≈ true_hessian rtol = rtol atol = atol

    # F and H
    fill!(hessian, NaN)
    objective_hessian!(hessian, model, params)
    @test hessian ≈ true_hessian rtol = rtol atol = atol

    # G and H
    fill!(hessian, NaN)
    gradient_hessian!(gradient, hessian, model, params)
    @test hessian ≈ true_hessian rtol = rtol atol = atol

    # F, G and H
    fill!(hessian, NaN)
    objective_gradient_hessian!(gradient, hessian, model, params)
    @test hessian ≈ true_hessian rtol = rtol atol = atol
end

fitmeasure_names_ml = Dict(
    :AIC => "aic",
    :BIC => "bic",
    :dof => "df",
    :χ² => "chisq",
    :p_value => "pvalue",
    :nparams => "npar",
    :RMSEA => "rmsea",
)

fitmeasure_names_ls = Dict(
    :dof => "df",
    :χ² => "chisq",
    :p_value => "pvalue",
    :nparams => "npar",
    :RMSEA => "rmsea",
)

function test_fitmeasures(
    measures,
    measures_lav;
    rtol = 1e-4,
    atol = 0,
    fitmeasure_names = fitmeasure_names_ml,
)
    @testset "$name" for (key, name) in pairs(fitmeasure_names)
        measure_lav = measures_lav.x[findfirst(==(name), measures_lav[!, 1])]
        @test measures[key] ≈ measure_lav rtol = rtol atol = atol
    end
end

function test_estimates(
    partable::ParameterTable,
    partable_lav;
    rtol = 1e-10,
    atol = 0,
    col = :estimate,
    lav_col = :est,
    lav_group = nothing,
    skip::Bool = false,
)
    actual = StructuralEquationModels.params(partable, col)
    expected =
        StructuralEquationModels.lavaan_params(partable_lav, partable, lav_col, lav_group)
    @test !any(isnan, actual)
    @test !any(isnan, expected)

    if skip # workaround skip=false not supported in earlier versions
        @test actual ≈ expected rtol = rtol atol = atol norm = Base.Fix2(norm, Inf) skip =
            skip
    else
        @test actual ≈ expected rtol = rtol atol = atol norm = Base.Fix2(norm, Inf)
    end
end

function test_estimates(
    ens_partable::EnsembleParameterTable,
    partable_lav;
    rtol = 1e-10,
    atol = 0,
    col = :estimate,
    lav_col = :est,
    lav_groups::AbstractDict,
    skip::Bool = false,
)
    actual = fill(NaN, nparams(ens_partable))
    expected = fill(NaN, nparams(ens_partable))
    for (key, partable) in pairs(ens_partable.tables)
        StructuralEquationModels.params!(actual, partable, col)
        StructuralEquationModels.lavaan_params!(
            expected,
            partable_lav,
            partable,
            lav_col,
            lav_groups[key],
        )
    end
    @test !any(isnan, actual)
    @test !any(isnan, expected)

    if skip # workaround skip=false not supported in earlier versions
        @test actual ≈ expected rtol = rtol atol = atol norm = Base.Fix2(norm, Inf) skip =
            skip
    else
        @test actual ≈ expected rtol = rtol atol = atol norm = Base.Fix2(norm, Inf)
    end
end

function test_bootstrap(
    model_fit::SemFit;
    compare_hessian = true,
    rtol_hessian = 0.2,
    compare_bs = true,
    rtol_bs = 0.1,
    n_boot = 500,
    seed = 32432,
)
    @testset rng = Random.seed!(seed) "bootstrap" begin
        se_bs = @suppress se_bootstrap(model_fit; n_boot = n_boot)
        # hessian-based and bootstrap-based std.errors are close
        if compare_hessian
            se_he = @suppress se_hessian(model_fit)
            #println(maximum(abs.(se_he - se_bs)))
            @test isapprox(se_bs, se_he, rtol = rtol_hessian)
        end
        # se_bootstrap and bootstrap |> se are close
        if compare_bs
            bs_samples = bootstrap(model_fit; n_boot = n_boot)
            @test bs_samples.n_converged >= 0.95*n_boot
            bs_samples = reduce(hcat, bs_samples.samples[bs_samples.converged_mask])
            se_bs_2 = sqrt.(var(bs_samples, corrected = false, dims = 2))
            #println(maximum(abs.(se_bs_2 - se_bs)))
            @test isapprox(se_bs_2, se_bs, rtol = rtol_bs)
        end
    end
end

function smoketest_bootstrap(model_fit::SemFit; n_boot = 5)
    # just test that both methods succeed
    se_bs = se_bootstrap(model_fit; n_boot = n_boot)
    bs_samples = bootstrap(model_fit; n_boot = n_boot)
    return se_bs, bs_samples
end

function smoketest_CI_z(model_fit::SemFit, partable)
    se_he = @suppress se_hessian(model_fit)
    normal_CI!(partable, model_fit, se_he)
    z_test!(partable, model_fit, se_he)
end
