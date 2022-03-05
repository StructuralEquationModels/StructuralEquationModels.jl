# splice model and loss functions
start_fabin3(model::Union{Sem, SemForwardDiff, SemFiniteDiff}; kwargs...) = 
    start_fabin3(
        model.observed, 
        model.imply,
        model.diff, 
        model.loss.functions...,
        kwargs...)

# SemObsCommon
start_fabin3(observed::SemObsCommon, imply::Union{RAM, RAMSymbolic}, diff, args...; kwargs...) = 
    start_fabin3(
        imply.ram_matrices,
        observed.obs_cov,
        observed.obs_mean)

# SemObsMissing
function start_fabin3(observed::SemObsMissing, imply::Union{RAM, RAMSymbolic}, diff, args...; kwargs...)

    if !observed.em_model.fitted
        em_mvn(observed; kwargs...)
    end

    return start_fabin3(
        imply.ram_matrices,
        observed.em_model.Σ,
        observed.em_model.μ)
end


function start_fabin3(ram_matrices::RAMMatrices, Σ, μ)

    A, S, F, M, parameters = ram_matrices.A, ram_matrices.S, ram_matrices.F, ram_matrices.M, ram_matrices.parameters

    n_latent = size(F, 2) - size(F, 1)
    Fmat = Matrix(F)
    ind_observed = [any(isone.(Fmat[:, i])) for i in 1:size(F, 2)]
    n_var = size(F, 1)
    parameters = [parameters...]
    n_par = size(parameters, 1)

    # loading Matrix
    Λ = A[ind_observed, .!ind_observed]
    ind_observed = findall(ind_observed)

    # check in which matrix each parameter appears
    in_S = zeros(Bool, n_par)
    in_A = zeros(Bool, n_par)
    in_Λ = zeros(Bool, n_par)
    indices = Vector{CartesianIndex}(undef, n_par)

    start_val = zeros(n_par)

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

    if !isnothing(M)
        in_M = zeros(Bool, n_par)
        for (i, par) ∈ enumerate(parameters)
            for j in CartesianIndices(M)
                if isequal(par, M[j])
                    in_M[i] = true
                    indices[i] = j
                end
            end
        end
    end

    if !isnothing(M) 
        in_any = in_A .| in_S .| in_M
    else 
        in_any = in_A .| in_S 
    end

    if !all(in_any)
        @warn "Could not determine starting value for the $i-th parameter. Default to 0"
    end

    # set undirected parameters in S
    for (par, in_A, in_S, in_Λ, index, i) ∈ zip(parameters, in_A, in_S, in_Λ, indices, 1:n_par)
        if in_S
            if index[1] == index[2]
                if index[1] ∈ ind_observed
                    index_position = findall(index[1] .== ind_observed)
                    start_val[i] = Σ[index_position[1], index_position[1]]/2
                else
                    start_val[i] = 0.05
                end
            end
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

    # set means
    for (in_M, index, i) ∈ zip(in_M, indices, 1:n_par)
        if in_M
            start_val[i] = μ[index]
        end
    end

    return start_val
end