Base.@kwdef struct RAMMatrices
    A
    S
    F
    M = nothing
    parameters
end

############################################################################
### get RAMMatrices from parameter table
############################################################################

function RAMMatrices(partable; parname = :θ, to_sparse = true)
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

    return RAMMatrices(;A = A, S = S, F = F, parameters = parameters)
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

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, ram_matrices::RAMMatrices)
    print_type_name(io, ram_matrices)
    print_field_types(io, ram_matrices)
end