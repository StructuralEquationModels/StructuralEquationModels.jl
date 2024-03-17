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
    true_hessian = FiniteDiff.finite_difference_hessian(x -> objective!(model, x)[1], parameters)
    hessian = zeros(size(true_hessian)); hessian .= 1.0
    gradient = similar(parameters)

    # H
    hessian!(hessian, model, parameters)
    correct1 = isapprox(hessian, true_hessian; rtol = rtol, atol = atol)

    # F and H
    hessian .= 1.0
    objective_hessian!(hessian, model, parameters)
    correct2 = isapprox(hessian, true_hessian; rtol = rtol, atol = atol)

    # G and H
    hessian .= 1.0
    gradient_hessian!(gradient, hessian, model, parameters)
    correct3 = isapprox(hessian, true_hessian; rtol = rtol, atol = atol)

    # F, G and H
    hessian .= 1.0
    objective_gradient_hessian!(gradient, hessian, model, parameters)
    correct4 = isapprox(hessian, true_hessian; rtol = rtol, atol = atol)
    
    return correct1 & correct2 & correct3 & correct4
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
    correct = []
    for key in keys(fitmeasure_names)
        measure = measures[key]
        measure_lav = measures_lav.x[measures_lav[:, 1] .==  fitmeasure_names[key]][1]
        push!(correct, isapprox(measure, measure_lav; rtol = rtol, atol = atol))
    end
    return correct
end

function compare_estimates(partable::ParameterTable, partable_lav;
        rtol = 1e-10, atol = 0, col = :estimate, lav_col = :est)

    correct = []

    for i in findall(partable.columns[:free])

        from = partable.columns[:from][i]
        to = partable.columns[:to][i]
        type = partable.columns[:parameter_type][i]
        estimate = partable.columns[col][i]

        if from == Symbol("1")

            lav_ind = findall(
                (partable_lav.lhs .== String(to)) .& 
                (partable_lav.op .== "~1"))

            if length(lav_ind) == 0
                throw(ErrorException("Parameter from: $from, to: $to, type: $type, could not be found in the lavaan solution"))
            elseif length(lav_ind) > 1
                throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
            else
                is_correct = isapprox(
                    estimate, 
                    partable_lav[:, lav_col][lav_ind[1]]; 
                    rtol = rtol, 
                    atol = atol)
                push!(correct, is_correct)
            end

        else
            
            if type == :↔
                type = "~~"
            elseif type == :→
                if (from ∈ partable.variables.latent) && (to ∈ partable.variables.observed)
                    type = "=~"
                else
                    type = "~"
                    from, to = to, from
                end
            end

            if type == "~~"

                lav_ind = findall(
                    (
                        ((partable_lav.lhs .== String(from)) .& (partable_lav.rhs .== String(to))) .|
                        ((partable_lav.lhs .== String(to)) .& (partable_lav.rhs .== String(from)))
                    ) .&
                    (partable_lav.op .== type)
                )

                if length(lav_ind) == 0
                    throw(ErrorException("Parameter from: $from, to: $to, type: $type, could not be found in the lavaan solution"))
                elseif length(lav_ind) > 1
                    throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
                else
                    is_correct = isapprox(
                        estimate, 
                        partable_lav[:, lav_col][lav_ind[1]]; 
                        rtol = rtol, 
                        atol = atol)
                    push!(correct, is_correct)
                end

            else
                lav_ind = findall(
                    (partable_lav.lhs .== String(from)) .& 
                    (partable_lav.rhs .== String(to)) .&
                    (partable_lav.op .== type))

                if length(lav_ind) == 0
                    throw(ErrorException("Parameter from: $from, to: $to, type: $type, could not be found in the lavaan solution"))
                elseif length(lav_ind) > 1
                    throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
                else
                    is_correct = isapprox(estimate, partable_lav[:, lav_col][lav_ind[1]]; rtol = rtol, atol = atol)
                    push!(correct, is_correct)
                end
            end

        end

    end
    
    return all(correct)
end

function compare_estimates(ens_partable::EnsembleParameterTable, partable_lav;
    rtol = 1e-10, atol = 0, col = :estimate, lav_col = :est,
    lav_groups)

    correct = []

    for key in keys(ens_partable.tables)

        group = lav_groups[key]
        partable = ens_partable.tables[key]

        for i in findall(partable.columns[:free])

            from = partable.columns[:from][i]
            to = partable.columns[:to][i]
            type = partable.columns[:parameter_type][i]
            estimate = partable.columns[col][i]

            if from == Symbol("1")

                lav_ind = findall(
                    (partable_lav.lhs .== String(to)) .& 
                    (partable_lav.op .== "~1") .&
                    (partable_lav.group .== group))

                if length(lav_ind) == 0
                    throw(ErrorException("Mean parameter of variable $to could not be found in the lavaan solution"))
                elseif length(lav_ind) > 1
                    throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
                else
                    is_correct = isapprox(
                        estimate, 
                        partable_lav[:, lav_col][lav_ind[1]]; 
                        rtol = rtol, 
                        atol = atol)
                    push!(correct, is_correct)
                end

            else
                
                if type == :↔
                    type = "~~"
                elseif type == :→
                    if (from ∈ partable.variables.latent) && (to ∈ partable.variables.observed)
                        type = "=~"
                    else
                        type = "~"
                        from, to = to, from
                    end
                end

                if type == "~~"

                    lav_ind = findall(
                        (
                            ((partable_lav.lhs .== String(from)) .& (partable_lav.rhs .== String(to))) .|
                            ((partable_lav.lhs .== String(to)) .& (partable_lav.rhs .== String(from)))
                        ) .&
                        (partable_lav.op .== type) .&
                        (partable_lav.group .== group)
                    )
    
                    if length(lav_ind) == 0
                        throw(ErrorException("Parameter from: $from, to: $to, type: $type, could not be found in the lavaan solution"))
                    elseif length(lav_ind) > 1
                        throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
                    else
                        is_correct = isapprox(
                            estimate, 
                            partable_lav[:, lav_col][lav_ind[1]]; 
                            rtol = rtol, 
                            atol = atol)
                        push!(correct, is_correct)
                    end
    
                else

                    lav_ind = findall(
                        (partable_lav.lhs .== String(from)) .& 
                        (partable_lav.rhs .== String(to)) .&
                        (partable_lav.op .== type).&
                        (partable_lav.group .== group))

                    if length(lav_ind) == 0
                        throw(ErrorException("Parameter $from $type $to could not be found in the lavaan solution"))
                    elseif length(lav_ind) > 1
                        throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
                    else
                        is_correct = isapprox(
                            estimate, 
                            partable_lav[:, lav_col][lav_ind[1]]; 
                            rtol = rtol, 
                            atol = atol)
                        push!(correct, is_correct)
                    end
                end

            end

        end

    end

    return all(correct)
end