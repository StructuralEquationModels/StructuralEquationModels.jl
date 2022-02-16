function get_parameter_type(string)
    if occursin("~~", string)
        return "~~" 
    elseif occursin("=~", string)
        return "=~"
    elseif occursin("~", string)
        return "~" 
    else
        return nothing
    end
end

function strip_nopar(string_vec)
    parameter_type = get_parameter_type.(string_vec)
    is_par = .!isnothing.(parameter_type)

    string_vec = string_vec[is_par]
    parameter_type = parameter_type[is_par]

    return (string_vec, parameter_type)
end

function expand_model_line(string, parameter_type)
    from, to = split(string, parameter_type)
    to = split(to, "+")

    from = remove_all_whitespace(from)
    to = remove_all_whitespace.(to)

    free = check_free.(to)
    value_fixed = get_fixed_value.(to, free)
    label = get_label.(to, free)

    from = fill(from, size(to, 1))
    to = last.(split.(to, "*"))

    if parameter_type == "=~"
        from, to = copy(to), copy(from)
        parameter_type = "~"
    end

    parameter_type = fill(parameter_type, size(to, 1))

    return from, parameter_type, to, free, value_fixed, label
end

remove_all_whitespace(string) = replace(string, r"\s" => "")

function get_partable(model_vec, parameter_type_in)

    from = Vector{String}()
    to = Vector{String}()
    parameter_type_out = Vector{String}()
    free = Vector{Bool}()
    value_fixed = Vector{Float64}()
    label = Vector{String}()

    for (model_line, parameter_type) in zip(model_vec, parameter_type_in)

        from_new, parameter_type_new, to_new, free_new, value_fixed_new, label_new = 
            expand_model_line(model_line, parameter_type)

        from = vcat(from, from_new)
        parameter_type_out = vcat(parameter_type_out, parameter_type_new)
        to = vcat(to, to_new)
        free = vcat(free, free_new)
        label = vcat(label, label_new)
        value_fixed = vcat(value_fixed, value_fixed_new)

    end

    start = copy(value_fixed)
    estimate = copy(value_fixed)

    return from, parameter_type_out, to, free, value_fixed, label, start, estimate
end

function check_free(to)
    if !occursin("*", to) 
        return true
    else
        fact = split(to, "*")[1]
        fact = remove_all_whitespace(fact)
        if check_str_number(fact)
            return false
        else
            return true
        end
    end
end

function get_label(to, free)
    if !occursin("*", to) || !free
        return ""
    else
        label = split(to, "*")[1]
        return label
    end
end

function get_fixed_value(to, free)
    if free
        return 0.0
    else
        fact = split(to, "*")[1]
        fact = remove_all_whitespace(fact)
        fact = parse(Float64, fact)
        return fact
    end
end

function check_str_number(string)
    return tryparse(Float64, string) !== nothing
end

function parse_sem(model)
    model_vec = split(model, "\n")
    model_vec, parameter_type = strip_nopar(model_vec)
    return get_partable(model_vec, parameter_type)
end



function get_RAM(partable, parname; to_sparse = true)
    n_labels_unique = size(unique(partable.label), 1) - 1
    n_labels = sum(.!(partable.label .== ""))
    n_parameters = sum(partable.free) - n_labels + n_labels_unique

    parameters = (Symbolics.@variables $parname[1:n_parameters])[1]

    n_observed = size(partable.observed_vars, 1)
    n_latent = size(partable.latent_vars, 1)
    n_node = n_observed + n_latent

    A = zeros(Num, n_node, n_node)
    S = zeros(Num, n_node, n_node)
    F = zeros(Num, n_observed, n_node)
    F[LinearAlgebra.diagind(F)] .= 1.0

    positions = Dict(zip([partable.observed_vars; partable.latent_vars], collect(1:n_observed+n_latent)))
    
    # fill Matrices
    known_labels = Dict{String, Int64}()
    par_ind = 1

    for i in 1:length(partable)

        from, parameter_type, to, free, value_fixed, label = partable[i]

        row_ind = positions[from]
        col_ind = positions[to]

        if !free
            if parameter_type == "~"
                A[row_ind, col_ind] = value_fixed
            else
                S[row_ind, col_ind] = value_fixed
                S[col_ind, row_ind] = value_fixed
            end
        else
            if label == ""
                set_RAM_index(A, S, parameter_type, row_ind, col_ind, parameters[par_ind])
                par_ind += 1
            else
                if haskey(known_labels, label)
                    known_ind = known_labels["label"]
                    set_RAM_index(A, S, parameter_type, row_ind, col_ind, parameters[known_ind])
                else
                    known_labels[label] = par_ind
                    set_RAM_index(A, S, parameter_type, row_ind, col_ind, parameters[par_ind])
                    par_ind +=1
                end
            end
        end

    end

    if to_sparse
        A = sparse(A)
        S = sparse(S)
        F = sparse(F)
    end

    return A, S, F, parameters
end

function set_RAM_index(A, S, parameter_type, row_ind, col_ind, parameter)
    if parameter_type == "~"
        A[row_ind, col_ind] = parameter
    else
        S[row_ind, col_ind] = parameter
        S[col_ind, row_ind] = parameter
    end
end