Base.@kwdef struct RAMMatrices
    A
    S
    F
    M = nothing
    parameters
    identifier = nothing
end

############################################################################
### get RAMMatrices from parameter table
############################################################################

function RAMMatrices!(partable::ParameterTable; parname::Symbol = :θ, to_sparse = true)
    n_labels_unique = size(unique(partable.label), 1) - 1
    n_labels = sum(.!(partable.label .== ""))
    n_parameters = sum(partable.free) - n_labels + n_labels_unique

    parameters = (Symbolics.@variables $parname[1:n_parameters])[1]
    
    identifier = Symbol.(parname, :_, 1:n_parameters)
    identifier_copy = copy(identifier)
    label_identifier = Dict{String, Symbol}()
    identifier_long = Vector{Symbol}()

    for label in partable.label[partable.free]
        if length(label) == 0
            push!(identifier_long, popfirst!(identifier_copy))
        else
            if haskey(label_identifier, label)
                push!(identifier_long, label_identifier[label])
            else
                push!(label_identifier, label => first(identifier_copy))
                push!(identifier_long, popfirst!(identifier_copy))
            end
        end
    end

    partable.identifier[partable.free] .= identifier_long

    n_observed = size(partable.observed_vars, 1)
    n_latent = size(partable.latent_vars, 1)
    n_node = n_observed + n_latent

    A = zeros(Num, n_node, n_node)
    S = zeros(Num, n_node, n_node)
    F = zeros(Num, n_observed, n_node)

    if length(partable.sorted_vars) != 0
        obsind = findall(x -> x ∈ partable.observed_vars, partable.sorted_vars)
        F[CartesianIndex.(1:n_observed, obsind)] .= 1.0
    else
        F[LinearAlgebra.diagind(F)] .= 1.0
    end

    if length(partable.sorted_vars) != 0
        positions = Dict(zip(partable.sorted_vars, collect(1:n_observed+n_latent)))
    else
        positions = Dict(zip([partable.observed_vars; partable.latent_vars], collect(1:n_observed+n_latent)))
    end
    
    # fill Matrices
    known_labels = Dict{String, Int64}()
    par_ind = 1

    for i in 1:length(partable)

        from, parameter_type, to, free, value_fixed, label = partable[i]

        row_ind = positions[to]
        col_ind = positions[from]

        if !free
            if parameter_type == "→"
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
                    known_ind = known_labels[label]
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

    return RAMMatrices(;A = A, S = S, F = F, parameters = parameters, identifier = identifier)
end

function set_RAM_index(A, S, parameter_type, row_ind, col_ind, parameter)
    if parameter_type == "→"
        A[row_ind, col_ind] = parameter
    else
        S[row_ind, col_ind] = parameter
        S[col_ind, row_ind] = parameter
    end
end

############################################################################
### get parameter table from RAMMatrices
############################################################################

function ParameterTable(;ram_matrices::RAMMatrices, colnames, kwargs...)
    
    new_partable = ParameterTable()

    # n_obs = size(ram_matrices.F, 1)
    # n_nod = size(ram_matrices.F, 2)
    # n_lat = n_nod - n_obs

    F = Matrix(ram_matrices.F)
    A = Matrix(ram_matrices.A)
    S = Matrix(ram_matrices.S)

    A_string = string.(A)
    S_string = string.(S)

    A_isfloat = check_str_number.(A_string)
    S_isfloat = check_str_number.(S_string)

    position_names = Dict{Int64, String}(1:length(colnames) .=> colnames)

    names_lat = Vector{String}()
    names_obs = Vector{String}()

    for (i, varname) in enumerate(colnames)
        if any(isone.(F[:, i]))
            push!(names_obs, varname)
        else
            push!(names_lat, varname)
        end
    end

    new_partable.observed_vars = copy(names_obs)
    new_partable.latent_vars = copy(names_lat)

    parameter_identifier = Dict([ram_matrices.parameters...] .=> ram_matrices.identifier)

    for index in CartesianIndices(A)
        value = A[index]
        if iszero(value)
        elseif A_isfloat[index]
            from = position_names[index[2]]
            to = position_names[index[1]]
            parameter_type = "→"
            free = false
            value_fixed = tryparse(Float64, A_string[index])
            label = "const"
            start = 0.0
            estimate = 0.0
            identifier = :const
            push!(new_partable, from, parameter_type, to, free, value_fixed, label, start, estimate, identifier)
        else
            from = position_names[index[2]]
            to = position_names[index[1]]
            parameter_type = "→"
            free = true
            value_fixed = 0.0
            label = string(parameter_identifier[value])
            start = 0.0
            estimate = 0.0
            identifier = parameter_identifier[value]
            push!(new_partable, from, parameter_type, to, free, value_fixed, label, start, estimate, identifier)
        end
    end

    for index in CartesianIndices(S)
        value = S[index]
        if iszero(value)
        elseif S_isfloat[index]
            from = position_names[index[2]]
            to = position_names[index[1]]
            parameter_type = "↔"
            free = false
            value_fixed = tryparse(Float64, S_string[index])
            label = "const"
            start = 0.0
            estimate = 0.0
            identifier = :const
            push!(new_partable, from, parameter_type, to, free, value_fixed, label, start, estimate, identifier)
        else
            from = position_names[index[2]]
            to = position_names[index[1]]
            parameter_type = "↔"
            free = true
            value_fixed = 0.0
            label = string(parameter_identifier[value])
            start = 0.0
            estimate = 0.0
            identifier = parameter_identifier[value]
            push!(new_partable, from, parameter_type, to, free, value_fixed, label, start, estimate, identifier)
        end
    end

    label_position = Dict{String, Vector{Int64}}()

    for (i, label) in enumerate(new_partable.label)
        if haskey(label_position, label)
            push!(label_position[label], i)
        else
            push!(label_position, label => [i])
        end
    end

    counter = 1

    for label in keys(label_position)
        if label == "const"
            new_partable.label[label_position[label]] .= ""
        elseif length(label_position[label]) == 1
            new_partable.label[label_position[label]] .= ""
        else
            new_partable.label[label_position[label]] .= "label_"*string(counter)
            counter += 1
        end
    end

    return new_partable
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, ram_matrices::RAMMatrices)
    print_type_name(io, ram_matrices)
    print_field_types(io, ram_matrices)
end