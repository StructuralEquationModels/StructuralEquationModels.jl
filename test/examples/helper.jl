using LinearAlgebra: norm

function test_gradient(model, parameters; rtol = 1e-10, atol = 0)
    true_grad = FiniteDiff.finite_difference_gradient(Base.Fix1(objective!, model), parameters)
    gradient = similar(parameters)

    # F and G
    fill!(gradient, NaN)
    gradient!(gradient, model, parameters)
    @test gradient ≈ true_grad rtol = rtol atol = atol

    # only G
    fill!(gradient, NaN)
    objective_gradient!(gradient, model, parameters)
    @test gradient ≈ true_grad rtol = rtol atol = atol
end

function test_hessian(model, parameters; rtol = 1e-4, atol = 0)
    true_hessian = FiniteDiff.finite_difference_hessian(Base.Fix1(objective!, model), parameters)
    hessian = similar(parameters, size(true_hessian))
    gradient = similar(parameters)

    # H
    fill!(hessian, NaN)
    hessian!(hessian, model, parameters)
    @test hessian ≈ true_hessian rtol = rtol atol = atol

    # F and H
    fill!(hessian, NaN)
    objective_hessian!(hessian, model, parameters)
    @test hessian ≈ true_hessian rtol = rtol atol = atol

    # G and H
    fill!(hessian, NaN)
    gradient_hessian!(gradient, hessian, model, parameters)
    @test hessian ≈ true_hessian rtol = rtol atol = atol

    # F, G and H
    fill!(hessian, NaN)
    objective_gradient_hessian!(gradient, hessian, model, parameters)
    @test hessian ≈ true_hessian rtol = rtol atol = atol
end

fitmeasure_names_ml = Dict(
    :AIC => "aic",
    :BIC => "bic",
    :df => "df",
    :χ² => "chisq",
    :p_value => "pvalue",
    :nparams => "npar",
    :RMSEA => "rmsea",
)

fitmeasure_names_ls = Dict(
    :df => "df",
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
        fitmeasure_names = fitmeasure_names_ml)

    @testset "$name" for (key, name) in pairs(fitmeasure_names)
            measure_lav = measures_lav.x[findfirst(==(name), measures_lav[!, 1])]
            @test measures[key] ≈ measure_lav rtol = rtol atol = atol
    end
end

function test_estimates(partable::ParameterTable, partable_lav;
        rtol = 1e-10, atol = 0, col = :estimate,
        lav_col = :est, lav_group = nothing,
        skip::Bool = false)

    actual = SEM.param_values(partable, col)
    expected = SEM.lavaan_param_values(partable_lav, partable, lav_col, lav_group)
    @test !any(isnan, actual)
    @test !any(isnan, expected)

    if skip # workaround skip=false not supported in earlier versions
        @test actual ≈ expected rtol = rtol atol = atol norm=Base.Fix2(norm, Inf) skip = skip
    else
        @test actual ≈ expected rtol = rtol atol = atol norm=Base.Fix2(norm, Inf)
    end
end

function test_estimates(ens_partable::EnsembleParameterTable, partable_lav;
    rtol = 1e-10, atol = 0, col = :estimate, lav_col = :est,
    lav_groups::AbstractDict, skip::Bool = false)

    actual = fill(NaN, nparams(ens_partable))
    expected = fill(NaN, nparams(ens_partable))
    for (key, partable) in pairs(ens_partable.tables)
        SEM.param_values!(actual, partable, col)
        SEM.lavaan_param_values!(expected, partable_lav, partable, lav_col, lav_groups[key])
    end
    @test !any(isnan, actual)
    @test !any(isnan, expected)

    if skip # workaround skip=false not supported in earlier versions
        @test actual ≈ expected rtol = rtol atol = atol norm=Base.Fix2(norm, Inf) skip = skip
    else
        @test actual ≈ expected rtol = rtol atol = atol norm=Base.Fix2(norm, Inf)
    end
end