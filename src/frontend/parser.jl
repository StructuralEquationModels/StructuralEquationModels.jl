function get_parameter_type(string)
    if occursin("∼∼", string)
        return "∼∼" 
    elseif occursin("=∼", string)
        return "=∼"
    elseif occursin("∼", string)
        return "∼" 
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

    if parameter_type == "=∼"
        from, to = copy(to), copy(from)
        parameter_type = "∼"
    end

    parameter_type = fill(parameter_type, size(to, 1))

    return convert(Vector{String}, from), 
        convert(Vector{String}, parameter_type), 
        convert(Vector{String}, to), free, value_fixed, 
        convert(Vector{String}, label)
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

    parameter_type_out = replace(parameter_type_out, "∼" => "→")
    parameter_type_out = replace(parameter_type_out, "∼∼" => "↔")

    return to, parameter_type_out, from, free, value_fixed, label, start, estimate
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

ParameterTable(lat_vars, obs_vars, model) = ParameterTable(lat_vars, obs_vars, Vector{String}(), parse_sem(model)...)