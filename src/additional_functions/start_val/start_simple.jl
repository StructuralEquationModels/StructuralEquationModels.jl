function start_simple(model::Union{Sem, SemForwardDiff, SemFiniteDiff}; kwargs...)
    return start_simple(
        model.observed, 
        model.imply,
        model.diff, 
        model.loss.functions...,
        kwargs...)
end

function start_simple(observed, imply::Union{RAM, RAMSymbolic}, diff, args...; kwargs...)
    return start_simple(imply.ram_matrices; kwargs...)
end

function start_simple(
    ram_matrices::RAMMatrices;
    start_loadings = 0.5,
    start_regressions = 0.0,
    start_variances_observed = 1,
    start_variances_latent = 0.05,
    start_covariances_observed = 0.0,
    start_covariances_latent = 0.0,
    start_covariances_obs_lat = 0.0,
    start_means = 0.0,
    kwargs...)

    A_ind, S_ind, F_ind, M_ind, parameters = 
        ram_matrices.A_ind, ram_matrices.S_ind, ram_matrices.F_ind, ram_matrices.M_ind, ram_matrices.parameters

    n_par = length(parameters)
    start_val = zeros(n_par)
    n_var, n_nod = ram_matrices.size_F

    C_indices = CartesianIndices((n_nod, n_nod))

    for i in 1:n_par
        if length(S_ind[i]) != 0
            # use the first occurence of the parameter to determine starting value
            c_ind = C_indices[S_ind[i][1]] 
            if c_ind[1] == c_ind[2]
                if c_ind[1] ∈ F_ind
                    start_val[i] = start_variances_observed
                else
                    start_val[i] = start_variances_latent
                end
            else
                o1 = c_ind[1] ∈ F_ind
                o2 = c_ind[2] ∈ F_ind
                if o1 & o2
                    start_val[i] = start_covariances_observed
                elseif !o1 & !o2
                    start_val[i] = start_covariances_latent
                else
                    start_val[i] = start_covariances_obs_lat
                end
            end
        elseif length(A_ind[i]) != 0
            c_ind = C_indices[A_ind[i][1]]
            if (c_ind[1] ∈ F_ind) & !(c_ind[2] ∈ F_ind)
                start_val[i] = start_loadings
            else
                start_val[i] = start_regressions
            end
        elseif !isnothing(M) && (length(M_ind[i]) != 0)
            start_val[i] = start_means
        end
    end
    return start_val
end