function sem_summary(sem_fit::SemFit; show_fitmeasures = false, color = :light_cyan, digits = 2)
    print("\n")
    println("Fitted Structural Equation Model")
    print("\n")
    printstyled("--------------------------------- Properties --------------------------------- \n"; color = color)
    print("\n")
    println("Optimization algorithm:      $(optimizer(sem_fit))")
    println("Convergence:                 $(convergence(sem_fit))")
    println("No. iterations/evaluations:  $(n_iterations(sem_fit))")
    print("\n")
    println("Number of parameters:        $(nparams(sem_fit))")
    println("Number of observations:      $(n_obs(sem_fit))")
    print("\n")
    printstyled("----------------------------------- Model ----------------------------------- \n"; color = color)
    print("\n")
    print(sem_fit.model)
    print("\n")
    if show_fitmeasures
        printstyled("--------------------------------- Fitmeasures --------------------------------- \n"; color = color)
        print("\n")
        a = fit_measures(sem_fit)
        goal_length = maximum(length.(string.(keys(a)))) + 4
        for k in keys(a)
            key_length = length(string(k))
            print(k); print(repeat(" ", goal_length - key_length)); print(round(a[k]; digits = 2)); print("\n")
        end
    end
    print("\n")
end


function sem_summary(partable::ParameterTable; color = :light_cyan, secondary_color = :light_yellow, digits = 2, show_variables = true)

    if show_variables
        print("\n")
        printstyled("--------------------------------- Variables --------------------------------- \n"; color = color)
        print("\n")
        printstyled("Latent variables:    "; color = color); for var in partable.variables[:latent_vars] print("$var ") end; print("\n")
        printstyled("Observed variables:  "; color = color); for var in partable.variables[:observed_vars] print("$var ") end; print("\n")
        if haskey(partable.variables, :sorted_vars) && (length(partable.variables[:sorted_vars]) > 0)
            printstyled("Sorted variables:    "; color = color); for var in partable.variables[:sorted_vars] print("$var ") end; print("\n")
        end
    end

    print("\n")
    printstyled("---------------------------- Parameter Estimates ----------------------------- \n"; color = color)
    print("\n")

    columns = keys(partable.columns)

    printstyled("Loadings: \n"; color = color)
    print("\n")

    sorted_columns = [:to, :estimate, :param, :value_fixed, :start]
    loading_columns = sort_partially(sorted_columns, columns)
    header_cols = copy(loading_columns)

    for var in partable.variables[:latent_vars]
        indicator_indices =
            findall(
                (partable.columns[:from] .== var) .&
                (partable.columns[:relation] .== :→) .&
                (partable.columns[:to] .∈ [partable.variables[:observed_vars]])
        )
        loading_array = reduce(hcat, check_round(partable.columns[c][indicator_indices]; digits = digits) for c in loading_columns)

        printstyled(var; color = secondary_color); print("\n")
        print("\n")
        pretty_table(loading_array; header = header_cols, tf = PrettyTables.tf_borderless, alignment = :l)
        print("\n")

    end

    printstyled("Directed Effects: \n"; color = color)

    regression_indices =
            findall(
                (partable.columns[:relation] .== :→) .&
                (
                    (
                        (partable.columns[:to] .∈ [partable.variables[:observed_vars]]) .&
                        (partable.columns[:from] .∈ [partable.variables[:observed_vars]])
                    ) .|
                    (
                        (partable.columns[:to] .∈ [partable.variables[:latent_vars]]) .&
                        (partable.columns[:from] .∈ [partable.variables[:observed_vars]])
                    ) .|
                    (
                        (partable.columns[:to] .∈ [partable.variables[:latent_vars]]) .&
                        (partable.columns[:from] .∈ [partable.variables[:latent_vars]])
                    )
                )
            )

    sorted_columns = [:from, :relation, :to, :estimate, :param, :value_fixed, :start]
    regression_columns = sort_partially(sorted_columns, columns)

    regression_array = reduce(hcat, check_round(partable.columns[c][regression_indices]; digits = digits) for c in regression_columns)
    regression_columns[2] = Symbol("")

    print("\n")
    pretty_table(regression_array; header = regression_columns, tf = PrettyTables.tf_borderless, alignment = :l)
    print("\n")

    printstyled("Variances: \n"; color = color)

    variance_indices =
            findall(
                (partable.columns[:relation] .== :↔) .&
                (partable.columns[:to] .== partable.columns[:from])
            )

    sorted_columns = [:from, :relation, :to, :estimate, :param, :value_fixed, :start]
    variance_columns = sort_partially(sorted_columns, columns)

    variance_array = reduce(hcat, check_round(partable.columns[c][variance_indices]; digits = digits) for c in variance_columns)
    variance_columns[2] = Symbol("")

    print("\n")
    pretty_table(variance_array; header = variance_columns, tf = PrettyTables.tf_borderless, alignment = :l)
    print("\n")

    printstyled("Covariances: \n"; color = color)

    variance_indices =
            findall(
                (partable.columns[:relation] .== :↔) .&
                (partable.columns[:to] .!= partable.columns[:from])
            )

    sorted_columns = [:from, :relation, :to, :estimate, :param, :value_fixed, :start]
    variance_columns = sort_partially(sorted_columns, columns)

    variance_array = reduce(hcat, check_round(partable.columns[c][variance_indices]; digits = digits) for c in variance_columns)
    variance_columns[2] = Symbol("")

    print("\n")
    pretty_table(variance_array; header = variance_columns, tf = PrettyTables.tf_borderless, alignment = :l)
    print("\n")

    mean_indices =
        findall(
            (partable.columns[:relation] .== :→) .&
            (partable.columns[:from] .== Symbol("1"))
        )

    if length(mean_indices) > 0

        printstyled("Means: \n"; color = color)

        sorted_columns = [:from, :relation, :to, :estimate, :param, :value_fixed, :start]
        variance_columns = sort_partially(sorted_columns, columns)

        variance_array = reduce(hcat, check_round(partable.columns[c][mean_indices]; digits = digits) for c in variance_columns)
        variance_columns[2] = Symbol("")

        print("\n")
        pretty_table(variance_array; header = variance_columns, tf = PrettyTables.tf_borderless, alignment = :l)
        print("\n")
    end

    #printstyled("""No need to copy and paste results, you can use CSV.write(DataFrame(my_partable), "myfile.csv")"""; hidden = true)

end


function sem_summary(partable::EnsembleParameterTable; color = :light_cyan, secondary_color = :light_yellow, digits = 2, show_variables = true)

    if show_variables
        print("\n")
        printstyled("--------------------------------- Variables --------------------------------- \n"; color = color)
        print("\n")
        let partable = partable.tables[[keys(partable.tables)...][1]]
            printstyled("Latent variables:    "; color = color); for var in partable.variables[:latent_vars] print("$var ") end; print("\n")
            printstyled("Observed variables:  "; color = color); for var in partable.variables[:observed_vars] print("$var ") end; print("\n")
            if haskey(partable.variables, :sorted_vars) && (length(partable.variables[:sorted_vars]) > 0)
                printstyled("Sorted variables:    "; color = color); for var in partable.variables[:sorted_vars] print("$var ") end; print("\n")
            end
        end
    end
    print("\n")

    for k in keys(partable.tables)
        print("\n")
        printstyled(rpad(" Group: $k", 78), reverse = true)
        print("\n")
        sem_summary(partable.tables[k]; color = color, secondary_color = secondary_color, digits = digits, show_variables = false)
    end

    # printstyled("""No need to copy and paste results, you can use CSV.write(DataFrame(my_partable), "myfile.csv")"""; hidden = true)

end

function check_round(vec; digits)
    if eltype(vec) <: Number
        return round.(vec; digits = digits)
    end
    return vec
end

function sort_partially(sorted, to_sort)
    out = Vector{eltype(to_sort)}()
    for el in sorted
        if el ∈ to_sort
            push!(out, el)
        end
    end
    remaining = setdiff(to_sort, sorted)
    append!(out, sort(collect(remaining)))
    return out
end

"""
    (1) sem_summary(sem_fit::SemFit; show_fitmeasures = false)

    (2) sem_summary(partable::AbstractParameterTable)

Print information about (1) a fitted SEM or (2) a parameter table to stdout.

# Extended help
## Addition keyword arguments
- `digits = 2`: controls precision of printed estimates, standard errors, etc.
- `color = :light_cyan`: color of some parts of the printed output. Can be adjusted for readability.
- `secondary_color = :light_yellow`
- `show_variables = true`
"""
function sem_summary end