############################################################################
### Type
############################################################################

struct RAMMatrices
    A_ind
    S_ind
    F_ind
    M_ind
    parameters
    colnames
    constants
    size_F
end

############################################################################
### Constructor
############################################################################

function RAMMatrices(;A, S, F, M = nothing, parameters, colnames)
    A_indices = get_parameter_indices(parameters, A)
    S_indices = get_parameter_indices(parameters, S)
    isnothing(M) ? M_indices = nothing : M_indices = get_parameter_indices(parameters, M)
    F_indices = findall([any(isone.(col)) for col in eachcol(F)])
    constants = get_RAMConstants(A, S, M)
    return RAMMatrices(A_indices, S_indices, F_indices, M_indices, parameters, colnames, constants, size(F))
end

############################################################################
### Constants
############################################################################

struct RAMConstant
    matrix
    index
    value
end

function get_RAMConstants(A, S, M)
    
    constants = Vector{RAMConstant}()

    for index in CartesianIndices(A)
        if (A[index] isa Number) && !iszero(A[index])
            push!(constants, RAMConstant(:A, index, A[index]))
        end
    end

    for index in CartesianIndices(S)
        if (S[index] isa Number) && !iszero(S[index])
            push!(constants, RAMConstant(:S, index, S[index]))
        end
    end

    if !isnothing(M)
        for index in CartesianIndices(M)
            if (M[index] isa Number) && !iszero(M[index])
                push!(constants, RAMConstant(:M, index, M[index]))
            end
        end
    end

    return constants

end

function set_RAMConstant!(A, S, M, rc::RAMConstant)
    if rc.matrix == :A
        A[rc.index] = rc.value
    elseif rc.matrix == :S
        S[rc.index] = rc.value
        S[rc.index[2], rc.index[1]] = rc.value
    elseif rc.matrix == :M
        M[rd.index] = rc.value
    end
end

function set_RAMConstants!(A, S, M, rc_vec::Vector{RAMConstant})
    for rc in rc_vec set_RAMConstant!(A, S, M, rc) end
end

############################################################################
### get RAMMatrices from parameter table
############################################################################

function RAMMatrices(partable::ParameterTable)

    parameters = unique(partable.identifier)
    filter!(x -> x != :const, parameters)
    n_par = length(parameters)
    par_positions = Dict(parameters .=> 1:n_par)

    n_observed = size(partable.observed_vars, 1)
    n_latent = size(partable.latent_vars, 1)
    n_node = n_observed + n_latent

    # F indices
    if length(partable.sorted_vars) != 0
        F_ind = findall(x -> x ∈ partable.observed_vars, partable.sorted_vars)
    else
        F_ind = 1:n_observed
    end

    # indices of the colnames
    if length(partable.sorted_vars) != 0
        positions = Dict(zip(partable.sorted_vars, collect(1:n_observed+n_latent)))
        colnames = copy(partable.sorted_vars)
    else
        positions = Dict(zip([partable.observed_vars; partable.latent_vars], collect(1:n_observed+n_latent)))
        colnames = [partable.observed_vars; partable.latent_vars]
    end
    
    # fill Matrices
    # known_labels = Dict{Symbol, Int64}()

    A_ind = Vector{Vector{Int64}}(undef, n_par)
    for i in 1:length(A_ind) A_ind[i] = Vector{Int64}() end
    S_ind = Vector{Vector{Int64}}(undef, n_par); S_ind .= [Vector{Int64}()]
    for i in 1:length(S_ind) S_ind[i] = Vector{Int64}() end

    # is there a meanstructure?
    if any(partable.from .== Symbol("1"))
        M_ind = Vector{Vector{Int64}}(undef, n_par)
        for i in 1:length(M_ind) M_ind[i] = Vector{Int64}() end
    else
        M_ind = nothing
    end

    # handel constants
    constants = Vector{RAMConstant}()
    
    for i in 1:length(partable)

        from, parameter_type, to, free, value_fixed, label, identifier = partable[i]

        row_ind = positions[to]
        if from != Symbol("1") col_ind = positions[from] end
        

        if !free
            if (parameter_type == :→) & (from == Symbol("1"))
                push!(constants, RAMConstant(:M, row_ind, value_fixed))
            elseif (parameter_type == :→)
                push!(constants, RAMConstant(:A, CartesianIndex(row_ind, col_ind), value_fixed))
            else
                push!(constants, RAMConstant(:S, CartesianIndex(row_ind, col_ind), value_fixed))
            end
        else
            par_ind = par_positions[identifier]
            if (parameter_type == :→) && (from == Symbol("1"))
                push!(M_ind[par_ind], row_ind)
            elseif parameter_type == :→
                push!(A_ind[par_ind], (row_ind + (col_ind-1)*n_node))
            else
                push!(S_ind[par_ind], row_ind + (col_ind-1)*n_node)
                if row_ind != col_ind
                    push!(S_ind[par_ind], col_ind + (row_ind-1)*n_node)
                end
            end
        end

    end

    return RAMMatrices(A_ind, S_ind, F_ind, M_ind, parameters, colnames, constants, (n_observed, n_node))
end

function set_RAM_index(A, S, parameter_type, row_ind, col_ind, parameter)
    if parameter_type == :→
        A[row_ind, col_ind] = parameter
    else
        S[row_ind, col_ind] = parameter
        S[col_ind, row_ind] = parameter
    end
end

############################################################################
### get parameter table from RAMMatrices
############################################################################

function ParameterTable(ram_matrices::RAMMatrices)
    
    new_partable = ParameterTable(nothing)

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

    position_names = Dict{Int64, String}(1:length(ram_matrices.colnames) .=> ram_matrices.colnames)

    names_lat = Vector{String}()
    names_obs = Vector{String}()

    for (i, varname) in enumerate(ram_matrices.colnames)
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