using LinearAlgebra: norm
using Suppressor

function is_extended_tests()
    return lowercase(get(ENV, "JULIA_EXTENDED_TESTS", "false")) == "true"
end

function test_gradient(model, params; rtol = 1e-10, atol = 0)
    @test nparams(model) == length(params)

    true_grad = FiniteDiff.finite_difference_gradient(Base.Fix1(objective!, model), params)

    gradient_G = fill!(similar(params), NaN)
    gradient!(gradient_G, model, params)
    gradient_FG = fill!(similar(params), NaN)
    objective_gradient!(gradient_FG, model, params)

    @test gradient_G == gradient_FG

    #@info "G norm = $(norm(gradient_G - true_grad, Inf))"
    @test gradient_G ≈ true_grad rtol = rtol atol = atol
end

function test_hessian(model, params; rtol = 1e-4, atol = 0)
    true_hessian =
        FiniteDiff.finite_difference_hessian(Base.Fix1(objective!, model), params)
    gradient = fill!(similar(params), NaN)

    hessian_H = fill!(similar(parent(true_hessian)), NaN)
    hessian!(hessian_H, model, params)

    hessian_FH = fill!(similar(hessian_H), NaN)
    objective_hessian!(hessian_FH, model, params)

    hessian_GH = fill!(similar(hessian_H), NaN)
    gradient_hessian!(gradient, hessian_GH, model, params)

    hessian_FGH = fill!(similar(hessian_H), NaN)
    objective_gradient_hessian!(gradient, hessian_FGH, model, params)

    @test hessian_H == hessian_FH == hessian_GH == hessian_FGH

    @test hessian_H ≈ true_hessian rtol = rtol atol = atol
end

# map from the SEM.jl name of the fit measure to the lavaan's one
fitmeasure_semjl_to_lavaan = Dict(
    :AIC => "aic",
    :BIC => "bic",
    :dof => "df",
    :χ² => "chisq",
    :p_value => "pvalue",
    :nparams => "npar",
    :RMSEA => "rmsea",
    :CFI => "cfi",
)

function test_fitmeasures(
    fitted::SemFit,
    measures_lav;
    fitmeasures::AbstractVector = SEM.DEFAULT_FIT_MEASURES,
    fitted_baseline::Union{SemFit, Nothing} = nothing,
    rtol = 1e-4,
    atol = 0,
)
    @testset "$fn" for fn in fitmeasures
        name = Symbol(fn)
        # FIML CFI requires the baseline model
        measure =
            fn != CFI || isnothing(fitted_baseline) ? fn(fitted) :
            fn(fitted, fitted_baseline)
        lav_name = fitmeasure_semjl_to_lavaan[name]
        lav_ix = findfirst(==(lav_name), measures_lav[!, 1])
        if isnothing(lav_ix)
            @test ismissing(measure)
        else
            measure_lav = measures_lav.x[lav_ix]
            @test measure ≈ measure_lav rtol = rtol atol = atol
        end
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
