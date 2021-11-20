###################### starting values FABIN 3
function start_fabin3(A, S, F, parameters, observed)
    n_latent = size(F, 2) - size(F, 1)
    n_var = size(F, 1)
    parameters = [parameters...]
    n_par = size(parameters, 1)

    # loading Matrix
    Λ = A[1:n_var, n_var+1:n_var+n_latent]

    # check in which matrix each parameter appears
    in_S = zeros(Bool, n_par)
    in_A = zeros(Bool, n_par)
    in_Λ = zeros(Bool, n_par)
    indices = Vector{CartesianIndex{2}}(undef, n_par)

    start_val = zeros(n_par)
    Σ = observed.obs_cov

    for (i, par) ∈ enumerate(parameters)
        for j in CartesianIndices(S)
            if isequal(par, S[j]) 
                in_S[i] = true
                indices[i] = j
            end
        end
        for j in CartesianIndices(A)
            if isequal(par, A[j]) 
                in_A[i] = true
                indices[i] = j 
            end 
        end
        for j in CartesianIndices(Λ)
            if isequal(par, Λ[j]) 
                in_Λ[i] = true
                indices[i] = j 
            end 
        end
    end

    # set undirected parameters in S
    for (par, in_A, in_S, in_Λ, index, i) ∈ zip(parameters, in_A, in_S, in_Λ, indices, 1:n_par)
        if in_S
            if index[1] == index[2]
                if index[1] <= n_var
                    start_val[i] = Σ[index]/2
                else
                    start_val[i] = 0.05
                end
            end
        elseif !in_A
            @warn "Could not determine starting value for the $i-th parameter. Default to 0"
        end
    end

    # set loadings
    for i ∈ 1:n_latent
        loadings = Λ[:, i]
        reference = findall(isone, loadings)
        indicators = findall(!iszero, loadings)

        # is there at least one reference indicator?
        if size(reference, 1) > 0
            if size(reference, 1) > 1
                @warn "You have more than 1 scaling indicator"
                reference = reference[1]
            else
                reference = reference[1]
            end

            for indicator in indicators
                if indicator != reference 
                    instruments = indicators[.!(indicators .∈ [[reference; indicator]])]

                    s32 = Σ[instruments, indicator]
                    s13 = Σ[reference, instruments]
                    S33 = Σ[instruments, instruments]
                    
                    if size(instruments, 1) == 1
                        temp = S33[1]/s13[1]
                        λ = s32[1]*temp/(s13[1]*temp)
                        start_val[isequal.(parameters, loadings[indicator])] .= λ
                    else
                        temp = S33\s13
                        λ = s32'*temp/(s13'*temp)
                        start_val[isequal.(parameters, loadings[indicator])] .= λ
                    end

                end
            end
        # no reference indicator:
        else
            reference = indicators[1]
            λ = zeros(size(indicators, 1)); λ[1] = 1.0
            for (j, indicator) in enumerate(indicators)
                if indicator != reference 
                    instruments = indicators[.!(indicators .∈ [[reference; indicator]])]

                    s32 = Σ[instruments, indicator]
                    s13 = Σ[reference, instruments]
                    S33 = Σ[instruments, instruments]
                    
                    if size(instruments, 1) == 1
                        temp = S33[1]/s13[1]
                        λ[j] = s32[1]*temp/(s13[1]*temp)
                    else
                        temp = S33\s13
                        λ[j] = s32'*temp/(s13'*temp)
                    end

                end
            end

            Σ_λ = Σ[indicators, indicators]
            D = λ*λ' ./ sum(λ.^2)
            θ = (I - D.^2)\(diag(Σ_λ - D*Σ_λ*D))
        
            # 3. psi
            Σ₁ = Σ_λ - Diagonal(θ)
            l₂ = sum(λ.^2)
            Ψ = sum( sum(λ.*Σ₁, dims = 1).*λ') ./ l₂^2
        
            λ = λ .* sign(Ψ) .* sqrt(abs(Ψ))

            for (j, indicator) ∈ enumerate(indicators)
                start_val[isequal.(parameters, loadings[indicator])] .= λ[j]
            end
        end
    end
    return start_val
end

function start_simple(A, S, F, parameters;
    M = nothing,
    loadings = 0.5,
    regressions = 0.0,
    variances_observed = 1,
    variances_latent = 0.05,
    covariances_observed = 0.0,
    covariances_latent = 0.0,
    means = 0.0)

    parameters = [parameters...]
    n_par = size(parameters, 1)
    start_val = zeros(n_par)
    n_var = size(F, 1)
    Λ_ind = filter(x -> (x[1] <= n_var) & (x[2] > n_var), CartesianIndices(A))

    for (i, par) ∈ enumerate(parameters)
        for index in CartesianIndices(S)
            if isequal(par, S[index]) 
                if index[1] == index[2]
                    if index[1] <= n_var
                        start_val[i] = variances_observed
                    else
                        start_val[i] = variances_latent
                    end
                else
                    if (index[1] <= n_var) & (index[1] <= n_var)
                        start_val[i] = covariances_observed
                    elseif (index[1] >= n_var) & (index[1] >= n_var)
                        start_val[i] = covariances_latent
                    end
                end
            end
        end
        for index in CartesianIndices(A)
            if isequal(par, A[index]) 
                if index ∈ Λ_ind
                    start_val[i] = loadings
                else
                    start_val[i] = regressions
                end
            end 
        end
        if !isnothing(M)
            for index in CartesianIndices(M)
                if isequal(par, M[index]) 
                    start_val[i] = means
                end 
            end
        end
    end

    return start_val
end