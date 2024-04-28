############################################################################################
### Type
############################################################################################

# map from parameter index to linear indices of matrix/vector positions where it occurs
AbstractArrayParamsMap = AbstractVector{<:AbstractVector{<:Integer}}
ArrayParamsMap = Vector{Vector{Int}}

struct RAMMatrices <: SemSpecification
    A_ind::ArrayParamsMap
    S_ind::ArrayParamsMap
    F_ind::Vector{Int}
    M_ind::Union{ArrayParamsMap, Nothing}
    parameters::Any
    colnames::Any
    constants::Any
    size_F::Any
end

############################################################################################
### Constructor
############################################################################################

function RAMMatrices(; A, S, F, M = nothing, parameters, colnames)
    A_indices = array_parameters_map(parameters, A)
    S_indices = array_parameters_map(parameters, S)
    M_indices = !isnothing(M) ? array_parameters_map(parameters, M) : nothing
    F_indices = findall([any(isone.(col)) for col in eachcol(F)])
    constants = get_RAMConstants(A, S, M)
    return RAMMatrices(
        A_indices,
        S_indices,
        F_indices,
        M_indices,
        parameters,
        colnames,
        constants,
        size(F),
    )
end

RAMMatrices(a::RAMMatrices) = a

############################################################################################
### Constants
############################################################################################

struct RAMConstant
    matrix::Any
    index::Any
    value::Any
end

import Base.==

function ==(c1::RAMConstant, c2::RAMConstant)
    res = ((c1.matrix == c2.matrix) && (c1.index == c2.index) && (c1.value == c2.value))
    return res
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
        M[rc.index] = rc.value
    end
end

function set_RAMConstants!(A, S, M, rc_vec::Vector{RAMConstant})
    for rc in rc_vec
        set_RAMConstant!(A, S, M, rc)
    end
end

############################################################################################
### get RAMMatrices from parameter table
############################################################################################

function RAMMatrices(partable::ParameterTable; par_id = nothing)
    if isnothing(par_id)
        parameters, n_par, par_positions = get_par_npar_identifier(partable)
    else
        parameters, n_par, par_positions =
            par_id[:parameters], par_id[:n_par], par_id[:par_positions]
    end

    n_observed = size(partable.variables[:observed_vars], 1)
    n_latent = size(partable.variables[:latent_vars], 1)
    n_node = n_observed + n_latent

    # F indices
    if length(partable.variables[:sorted_vars]) != 0
        F_ind = findall(
            x -> x ∈ partable.variables[:observed_vars],
            partable.variables[:sorted_vars],
        )
    else
        F_ind = 1:n_observed
    end

    # indices of the colnames
    if length(partable.variables[:sorted_vars]) != 0
        positions =
            Dict(zip(partable.variables[:sorted_vars], collect(1:n_observed+n_latent)))
        colnames = copy(partable.variables[:sorted_vars])
    else
        positions = Dict(
            zip(
                [partable.variables[:observed_vars]; partable.variables[:latent_vars]],
                collect(1:n_observed+n_latent),
            ),
        )
        colnames = [partable.variables[:observed_vars]; partable.variables[:latent_vars]]
    end

    # fill Matrices
    # known_labels = Dict{Symbol, Int64}()

    A_ind = Vector{Vector{Int64}}(undef, n_par)
    for i in 1:length(A_ind)
        A_ind[i] = Vector{Int64}()
    end
    S_ind = Vector{Vector{Int64}}(undef, n_par)
    S_ind .= [Vector{Int64}()]
    for i in 1:length(S_ind)
        S_ind[i] = Vector{Int64}()
    end

    # is there a meanstructure?
    if any(partable.columns[:from] .== Symbol("1"))
        M_ind = Vector{Vector{Int64}}(undef, n_par)
        for i in 1:length(M_ind)
            M_ind[i] = Vector{Int64}()
        end
    else
        M_ind = nothing
    end

    # handel constants
    constants = Vector{RAMConstant}()

    for i in 1:length(partable)
        from, parameter_type, to, free, value_fixed, identifier = partable[i]

        row_ind = positions[to]
        if from != Symbol("1")
            col_ind = positions[from]
        end

        if !free
            if (parameter_type == :→) & (from == Symbol("1"))
                push!(constants, RAMConstant(:M, row_ind, value_fixed))
            elseif (parameter_type == :→)
                push!(
                    constants,
                    RAMConstant(:A, CartesianIndex(row_ind, col_ind), value_fixed),
                )
            else
                push!(
                    constants,
                    RAMConstant(:S, CartesianIndex(row_ind, col_ind), value_fixed),
                )
            end
        else
            par_ind = par_positions[identifier]
            if (parameter_type == :→) && (from == Symbol("1"))
                push!(M_ind[par_ind], row_ind)
            elseif parameter_type == :→
                push!(A_ind[par_ind], (row_ind + (col_ind - 1) * n_node))
            else
                push!(S_ind[par_ind], row_ind + (col_ind - 1) * n_node)
                if row_ind != col_ind
                    push!(S_ind[par_ind], col_ind + (row_ind - 1) * n_node)
                end
            end
        end
    end

    return RAMMatrices(
        A_ind,
        S_ind,
        F_ind,
        M_ind,
        parameters,
        colnames,
        constants,
        (n_observed, n_node),
    )
end

############################################################################################
### get parameter table from RAMMatrices
############################################################################################

function ParameterTable(ram_matrices::RAMMatrices)
    partable = ParameterTable(nothing)

    colnames = ram_matrices.colnames
    position_names = Dict{Int64, Symbol}(1:length(colnames) .=> colnames)

    # observed and latent variables
    names_obs = colnames[ram_matrices.F_ind]
    names_lat = colnames[findall(x -> !(x ∈ ram_matrices.F_ind), 1:length(colnames))]

    partable.variables = Dict(
        :sorted_vars => Vector{Symbol}(),
        :observed_vars => names_obs,
        :latent_vars => names_lat,
    )

    # constants
    for c in ram_matrices.constants
        push!(partable, get_partable_row(c, position_names))
    end

    # parameters
    for (i, par) in enumerate(ram_matrices.parameters)
        push_partable_rows!(
            partable,
            position_names,
            par,
            i,
            ram_matrices.A_ind,
            ram_matrices.S_ind,
            ram_matrices.M_ind,
            ram_matrices.size_F[2],
        )
    end

    return partable
end

############################################################################################
### get RAMMatrices from EnsembleParameterTable
############################################################################################

function RAMMatrices(partable::EnsembleParameterTable)
    ram_matrices = Dict{Symbol, RAMMatrices}()

    parameters, n_par, par_positions = get_par_npar_identifier(partable)
    par_id =
        Dict(:parameters => parameters, :n_par => n_par, :par_positions => par_positions)

    for key in keys(partable.tables)
        ram_mat = RAMMatrices(partable.tables[key]; par_id = par_id)
        push!(ram_matrices, key => ram_mat)
    end

    return ram_matrices
end

############################################################################################
### Pretty Printing
############################################################################################

function Base.show(io::IO, ram_matrices::RAMMatrices)
    print_type_name(io, ram_matrices)
    print_field_types(io, ram_matrices)
end

############################################################################################
### Additional Functions
############################################################################################

function get_par_npar_identifier(partable::ParameterTable)
    parameters = unique(partable.columns[:identifier])
    filter!(x -> x != :const, parameters)
    n_par = length(parameters)
    par_positions = Dict(parameters .=> 1:n_par)
    return parameters, n_par, par_positions
end

function get_par_npar_identifier(partable::EnsembleParameterTable)
    parameters = Vector{Symbol}()
    for key in keys(partable.tables)
        append!(parameters, partable.tables[key].columns[:identifier])
    end
    parameters = unique(parameters)
    filter!(x -> x != :const, parameters)

    n_par = length(parameters)

    par_positions = Dict(parameters .=> 1:n_par)

    return parameters, n_par, par_positions
end

function get_partable_row(c::RAMConstant, position_names)
    # variable names
    from = position_names[c.index[2]]
    to = position_names[c.index[1]]
    # parameter type
    if c.matrix == :A
        parameter_type = :→
    elseif c.matrix == :S
        parameter_type = :↔
    elseif c.matrix == :M
        parameter_type = :→
    end
    free = false
    value_fixed = c.value
    start = 0.0
    estimate = 0.0
    identifier = :const
    return Dict(
        :from => from,
        :parameter_type => parameter_type,
        :to => to,
        :free => free,
        :value_fixed => value_fixed,
        :start => start,
        :estimate => estimate,
        :identifier => identifier,
    )
end

function cartesian_is_known(index, known_indices)
    known = false
    for k_in in known_indices
        if (index == k_in) | ((index[1] == k_in[2]) & (index[2] == k_in[1]))
            known = true
        end
    end
    return known
end

cartesian_is_known(index, known_indices::Nothing) = false

function get_partable_row(par, position_names, index, matrix, n_nod, known_indices)

    # variable names
    if matrix == :M
        from = Symbol("1")
        to = position_names[index]
    else
        index = linear2cartesian(index, (n_nod, n_nod))

        if (matrix == :S) & (cartesian_is_known(index, known_indices))
            return nothing
        elseif matrix == :S
            push!(known_indices, index)
        end

        from = position_names[index[2]]
        to = position_names[index[1]]
    end

    # parameter type
    if matrix == :A
        parameter_type = :→
    elseif matrix == :S
        parameter_type = :↔
    elseif matrix == :M
        parameter_type = :→
    end

    free = true
    value_fixed = 0.0
    start = 0.0
    estimate = 0.0
    identifier = par

    return Dict(
        :from => from,
        :parameter_type => parameter_type,
        :to => to,
        :free => free,
        :value_fixed => value_fixed,
        :start => start,
        :estimate => estimate,
        :identifier => identifier,
    )
end

function push_partable_rows!(partable, position_names, par, i, A_ind, S_ind, M_ind, n_nod)
    A_ind = A_ind[i]
    S_ind = S_ind[i]
    isnothing(M_ind) || (M_ind = M_ind[i])

    for ind in A_ind
        push!(partable, get_partable_row(par, position_names, ind, :A, n_nod, nothing))
    end

    known_indices = Vector{CartesianIndex}()
    for ind in S_ind
        push!(
            partable,
            get_partable_row(par, position_names, ind, :S, n_nod, known_indices),
        )
    end

    if !isnothing(M_ind)
        for ind in M_ind
            push!(partable, get_partable_row(par, position_names, ind, :M, n_nod, nothing))
        end
    end

    return nothing
end

function ==(mat1::RAMMatrices, mat2::RAMMatrices)
    res = (
        (mat1.A_ind == mat2.A_ind) &&
        (mat1.S_ind == mat2.S_ind) &&
        (mat1.F_ind == mat2.F_ind) &&
        (mat1.M_ind == mat2.M_ind) &&
        (mat1.parameters == mat2.parameters) &&
        (mat1.colnames == mat2.colnames) &&
        (mat1.size_F == mat2.size_F) &&
        (mat1.constants == mat2.constants)
    )
    return res
end

function get_group(d::Dict, group)
    return d[group]
end
