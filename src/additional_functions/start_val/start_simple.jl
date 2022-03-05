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
    start_means = 0.0,
    kwargs...)

    A, S, F, M, parameters = 
        ram_matrices.A, ram_matrices.S, ram_matrices.F, ram_matrices.M, ram_matrices.parameters

    parameters = [parameters...]
    n_par = size(parameters, 1)
    start_val = zeros(n_par)
    n_var = size(F, 1)

    Fmat = Matrix(F)
    ind_observed = [any(isone.(Fmat[:, i])) for i in 1:size(F, 2)]
    Λ_ind = CartesianIndices(A)[ind_observed, .!ind_observed]
    ind_observed = findall(ind_observed)

    for (i, par) ∈ enumerate(parameters)
        for index in CartesianIndices(S)
            if isequal(par, S[index])
                if index[1] == index[2]
                    if index[1] ∈ ind_observed
                        start_val[i] = start_variances_observed
                    else
                        start_val[i] = start_variances_latent
                    end
                else
                    if (index[1] <= n_var) & (index[1] <= n_var)
                        start_val[i] = start_covariances_observed
                    elseif (index[1] >= n_var) & (index[1] >= n_var)
                        start_val[i] = start_covariances_latent
                    end
                end
            end
        end
        for index in CartesianIndices(A)
            if isequal(par, A[index]) 
                if index ∈ Λ_ind
                    start_val[i] = start_loadings
                else
                    start_val[i] = start_regressions
                end
            end 
        end
        if !isnothing(M)
            for index in CartesianIndices(M)
                if isequal(par, M[index]) 
                    start_val[i] = start_means
                end 
            end
        end
    end

    return start_val
end