function details(sem::AbstractSem)
    print("Structural Equation Model")
    print(_subtype_info(sem))
    print("\n")
    print("- Loss Functions \n")
    for term in loss_terms(sem)
        print("  > ")
        details(term)
        println()
    end
end

function details(term::LossTerm)
    if !issemloss(term)
        print(term.loss)
    else
        println("Structural Equation Model Loss ($(nameof(typeof(term.loss))))")
        if !isnothing(id(term))
            print("    - id:          $(id(term)) \n")
        end
        println(
            "    - Observed:    $(nameof(typeof(observed(term)))) ($(nsamples(term)) samples)",
        )
        println(
            "    - Implied:     $(nameof(typeof(implied(term)))) ($(nparams(term)) parameters)",
        )
        println(
            "    - Variables:   $(nobserved_vars(term)) observed, $(nlatent_vars(term)) latent",
        )
        if !isnothing(weight(term))
            print("    - weight:      $(round(weight(term), digits=3))")
        end
    end
end

function details(sem_fit::SemFit; show_fitmeasures = false, color = :light_cyan, digits = 2)
    print("\n")
    println("Fitted Structural Equation Model")
    print("\n")
    printstyled(
        "--------------------------------- Properties --------------------------------- \n";
        color = color,
    )
    print("\n")
    println("Optimization engine:         $(optimizer_engine(sem_fit))")
    println("Optimization algorithm:      $(algorithm_name(sem_fit))")
    println("Convergence:                 $(convergence(sem_fit))")
    println("No. iterations/evaluations:  $(n_iterations(sem_fit))")
    print("\n")
    println("Number of parameters:        $(nparams(sem_fit))")
    println("Number of data samples:      $(nsamples(sem_fit))")
    print("\n")
    printstyled(
        "----------------------------------- Model ------------------------------------ \n";
        color = color,
    )
    print("\n")
    print(sem_fit.model)
    print("\n")
    if show_fitmeasures
        printstyled(
            "-------------------------------- Fitmeasures --------------------------------- \n";
            color = color,
        )
        print("\n")
        a = fit_measures(sem_fit)
        goal_length = maximum(length.(string.(keys(a)))) + 4
        for k in keys(a)
            key_length = length(string(k))
            print(k)
            print(repeat(" ", goal_length - key_length))
            print(round(a[k]; digits = 2))
            print("\n")
        end
    end
    print("\n")
end

function details(
    partable::ParameterTable;
    color = :light_cyan,
    secondary_color = :light_yellow,
    digits = 2,
    show_variables = true,
    show_columns = nothing,
)
    if show_variables
        print("\n")
        printstyled(
            "---------------------------------- Variables --------------------------------- \n";
            color = color,
        )
        print("\n")
        printstyled("Latent variables:    "; color = color)
        for var in partable.latent_vars
            print("$var ")
        end
        print("\n")
        printstyled("Observed variables:  "; color = color)
        for var in partable.observed_vars
            print("$var ")
        end
        print("\n")
        if length(partable.sorted_vars) > 0
            printstyled("Sorted variables:    "; color = color)
            for var in partable.sorted_vars
                print("$var ")
            end
            print("\n")
        end
    end

    print("\n")
    printstyled(
        "---------------------------- Parameter Estimates ----------------------------- \n";
        color = color,
    )
    print("\n")

    columns = keys(partable.columns)
    show_columns = isnothing(show_columns) ? nothing : intersect(show_columns, columns)

    printstyled("Loadings: \n"; color = color)
    print("\n")

    if isnothing(show_columns)
        sorted_columns = [:to, :estimate, :param, :value_fixed, :start]
        loading_columns = sort_partially(sorted_columns, columns)
        header_cols = copy(loading_columns)
    else
        loading_columns = copy(show_columns)
        header_cols = copy(loading_columns)
    end

    for var in partable.latent_vars
        indicator_indices = findall(
            r ->
                (r.from == var) && (r.relation == :→) && (r.to ∈ partable.observed_vars),
            partable,
        )
        loading_array = reduce(
            hcat,
            check_round(partable.columns[c][indicator_indices]; digits = digits) for
            c in loading_columns
        )

        printstyled(var; color = secondary_color)
        print("\n")
        print("\n")
        pretty_table(
            loading_array;
            column_labels = header_cols,
            table_format = TextTableFormat(borders = text_table_borders__borderless),
            alignment = :l,
            formatters = [(v, i, j) -> isa(v, Number) && isnan(v) ? "" : v],
        )
        print("\n")
    end

    printstyled("Directed Effects: \n"; color = color)

    regression_indices = findall(
        r ->
            (r.relation == :→) && (
                ((r.to ∈ partable.observed_vars) && (r.from ∈ partable.observed_vars)) ||
                ((r.to ∈ partable.latent_vars) && (r.from ∈ partable.observed_vars)) ||
                ((r.to ∈ partable.latent_vars) && (r.from ∈ partable.latent_vars))
            ),
        partable,
    )

    if isnothing(show_columns)
        sorted_columns = [:from, :relation, :to, :estimate, :param, :value_fixed, :start]
        regression_columns = sort_partially(sorted_columns, columns)
    else
        regression_columns = copy(show_columns)
    end

    regression_array = reduce(
        hcat,
        check_round(partable.columns[c][regression_indices]; digits = digits) for
        c in regression_columns
    )
    regression_columns[2] =
        regression_columns[2] == :relation ? Symbol("") : regression_columns[2]

    print("\n")
    pretty_table(
        regression_array;
        column_labels = regression_columns,
        table_format = TextTableFormat(borders = text_table_borders__borderless),
        alignment = :l,
        formatters = [(v, i, j) -> isa(v, Number) && isnan(v) ? "" : v],
    )
    print("\n")

    printstyled("Variances: \n"; color = color)

    var_indices = findall(r -> r.relation == :↔ && r.to == r.from, partable)

    if isnothing(show_columns)
        sorted_columns = [:from, :relation, :to, :estimate, :param, :value_fixed, :start]
        var_columns = sort_partially(sorted_columns, columns)
    else
        var_columns = copy(show_columns)
    end

    var_array = reduce(
        hcat,
        check_round(partable.columns[c][var_indices]; digits) for c in var_columns
    )
    var_columns[2] = var_columns[2] == :relation ? Symbol("") : var_columns[2]

    print("\n")
    pretty_table(
        var_array;
        column_labels = var_columns,
        table_format = TextTableFormat(borders = text_table_borders__borderless),
        alignment = :l,
        formatters = [(v, i, j) -> isa(v, Number) && isnan(v) ? "" : v],
    )
    print("\n")

    printstyled("Covariances: \n"; color = color)

    covar_indices = findall(r -> r.relation == :↔ && r.to != r.from, partable)

    if isnothing(show_columns)
        covar_columns = sort_partially(sorted_columns, columns)
    else
        covar_columns = copy(show_columns)
    end

    covar_array = reduce(
        hcat,
        check_round(partable.columns[c][covar_indices]; digits = digits) for
        c in covar_columns
    )
    covar_columns[2] = covar_columns[2] == :relation ? Symbol("") : covar_columns[2]

    print("\n")
    pretty_table(
        covar_array;
        column_labels = covar_columns,
        table_format = TextTableFormat(borders = text_table_borders__borderless),
        alignment = :l,
        formatters = [(v, i, j) -> isa(v, Number) && isnan(v) ? "" : v],
    )
    print("\n")

    mean_indices = findall(r -> (r.relation == :→) && (r.from == Symbol(1)), partable)

    if length(mean_indices) > 0
        printstyled("Means: \n"; color = color)

        if isnothing(show_columns)
            sorted_columns =
                [:from, :relation, :to, :estimate, :param, :value_fixed, :start]
            mean_columns = sort_partially(sorted_columns, columns)
        else
            mean_columns = copy(show_columns)
        end

        mean_array = reduce(
            hcat,
            check_round(partable.columns[c][mean_indices]; digits = digits) for
            c in mean_columns
        )
        mean_columns[2] = mean_columns[2] == :relation ? Symbol("") : mean_columns[2]

        print("\n")
        pretty_table(
            mean_array;
            column_labels = mean_columns,
            table_format = TextTableFormat(borders = text_table_borders__borderless),
            alignment = :l,
            formatters = [(v, i, j) -> isa(v, Number) && isnan(v) ? "" : v],
        )
        print("\n")
    end
end

function details(
    partable::EnsembleParameterTable;
    color = :light_cyan,
    secondary_color = :light_yellow,
    digits = 2,
    show_variables = true,
    show_columns = nothing,
)
    if show_variables
        print("\n")
        printstyled(
            "--------------------------------- Variables --------------------------------- \n";
            color = color,
        )
        print("\n")
        let partable = partable.tables[[keys(partable.tables)...][1]]
            printstyled("Latent variables:    "; color = color)
            for var in partable.latent_vars
                print("$var ")
            end
            print("\n")
            printstyled("Observed variables:  "; color = color)
            for var in partable.observed_vars
                print("$var ")
            end
            print("\n")
            if length(partable.sorted_vars) > 0
                printstyled("Sorted variables:    "; color = color)
                for var in partable.sorted_vars
                    print("$var ")
                end
                print("\n")
            end
        end
    end
    print("\n")

    for k in keys(partable.tables)
        print("\n")
        printstyled(rpad(" Group: $k", 78), reverse = true)
        print("\n")
        details(
            partable.tables[k];
            color = color,
            secondary_color = secondary_color,
            digits = digits,
            show_variables = false,
            show_columns = show_columns,
        )
    end
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

function Base.findall(fun::Function, partable::ParameterTable)
    rows = Int[]
    for (i, r) in enumerate(partable)
        fun(r) ? push!(rows, i) : nothing
    end
    return rows
end

"""
    (1) details(model::AbstractSem)

    (2) details(sem_fit::SemFit; show_fitmeasures = false)

    (3) details(partable::AbstractParameterTable; ...)

Print information about (1) a SEM, (2) a fitted SEM or (3) a parameter table to stdout.

# Extended help
## Addition keyword arguments
- `digits = 2`: controls precision of printed estimates, standard errors, etc.
- `color = :light_cyan`: color of some parts of the printed output. Can be adjusted for readability.
- `secondary_color = :light_yellow`
- `show_variables = true`
- `show_columns = nothing`: columns names to include in the output e.g.`[:from, :to, :estimate]`)
"""
function details end
