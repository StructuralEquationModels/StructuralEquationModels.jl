"""
    start_fabin3(model)
    
Return a vector of FABIN 3 starting values (see Hägglund 1982).
Not available for ensemble models.
"""
function start_fabin3 end

# splice model and loss functions
function start_fabin3(
        model::AbstractSemSingle; 
        kwargs...)
    return start_fabin3(
        model.observed, 
        model.imply,
        model.optimizer, 
        model.loss.functions...,
        kwargs...)
end

function start_fabin3(
        observed, 
        imply, 
        optimizer, 
        args...;
        kwargs...)
    return start_fabin3(
        ram_matrices(imply),
        obs_cov(observed),
        obs_mean(observed))
end

# SemObservedMissing
function start_fabin3(
        observed::SemObservedMissing, 
        imply, 
        optimizer, 
        args...; 
        kwargs...)

    if !observed.em_model.fitted
        em_mvn(observed; kwargs...)
    end

    return start_fabin3(
        ram_matrices(imply),
        observed.em_model.Σ,
        observed.em_model.μ)
end


function start_fabin3(ram_matrices::RAMMatrices, Σ, μ)

    A_ind, S_ind, F_ind, M_ind, parameters = 
        ram_matrices.A_ind, 
        ram_matrices.S_ind, 
        ram_matrices.F_ind, 
        ram_matrices.M_ind, 
        ram_matrices.parameters

    n_par = length(parameters)
    start_val = zeros(n_par)
    n_var, n_nod = ram_matrices.size_F
    n_latent = n_nod - n_var

    C_indices = CartesianIndices((n_nod, n_nod))

    # check in which matrix each parameter appears
    
    indices = Vector{CartesianIndex}(undef, n_par)

    start_val = zeros(n_par)

#=     in_S = length.(S_ind) .!= 0
    in_A = length.(A_ind) .!= 0
    A_ind_c = [linear2cartesian(ind, (n_nod, n_nod)) for ind in A_ind]
    in_Λ = [any(ind[2] .∈ F_ind) for ind in A_ind_c]

    if !isnothing(M)
        in_M = length.(M_ind) .!= 0
        in_any = in_A .| in_S .| in_M
    else 
        in_any = in_A .| in_S 
    end

    if !all(in_any)
        @warn "Could not determine fabin3 starting values for some parameters, default to 0."
    end =#

    # set undirected parameters in S
    for (i, S_ind) in enumerate(S_ind)
        if length(S_ind) != 0 
            c_ind = C_indices[S_ind][1]
            if c_ind[1] == c_ind[2]
                if c_ind[1] ∈ F_ind
                    index_position = findall(c_ind[1] .== F_ind)
                    start_val[i] = Σ[index_position[1], index_position[1]]/2
                else
                    start_val[i] = 0.05
                end
            end # covariances stay 0
        end
    end

    # set loadings
    constants = ram_matrices.constants
    A_ind_c = [linear2cartesian(ind, (n_nod, n_nod)) for ind in A_ind]
    # ind_Λ = findall([is_in_Λ(ind_vec, F_ind) for ind_vec in A_ind_c])

    for i ∈ findall(.!(1:n_nod .∈ [F_ind]))
        reference = Int64[]
        indicators = Int64[]
        loadings = Symbol[]

        for (j, ind_c_vec) in enumerate(A_ind_c)
            for ind_c in ind_c_vec
                if (ind_c[2] == i) & (ind_c[1] ∈ F_ind)
                    push!(indicators, ind_c[1])
                    push!(loadings, parameters[j])
                end
            end
        end

        for ram_constant in constants
            if (ram_constant.matrix == :A) & (ram_constant.index[2] == i) & (ram_constant.index[1] ∈ F_ind)
                push!(loadings, Symbol(""))
                if isone(ram_constant.value)
                    push!(reference, ram_constant.index[1])
                else
                    push!(indicators, ram_constant.index[1])
                end
            end
        end

        reference = [findfirst(x -> x == ref, F_ind) for ref in reference]
        indicators = [findfirst(x -> x == ind, F_ind) for ind in indicators]

        # is there at least one reference indicator?
        if size(reference, 1) > 0
            if size(reference, 1) > 1
                @warn "You have more than 1 scaling indicator"
                reference = reference[1]
            else
                reference = reference[1]
            end

            for (j, indicator) in enumerate(indicators)
                if indicator != reference 
                    instruments = indicators[.!(indicators .∈ [[reference; indicator]])]

                    s32 = Σ[instruments, indicator]
                    s13 = Σ[reference, instruments]
                    S33 = Σ[instruments, instruments]
                    
                    if size(instruments, 1) == 1
                        temp = S33[1]/s13[1]
                        λ = s32[1]*temp/(s13[1]*temp)
                        start_val[isequal.(parameters, loadings[j])] .= λ
                    else
                        temp = S33\s13
                        λ = s32'*temp/(s13'*temp)
                        start_val[isequal.(parameters, loadings[j])] .= λ
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
                start_val[isequal.(parameters, loadings[j])] .= λ[j]
            end
        end
    end

    # set means
    if !isnothing(M_ind)
        for (i, M_ind) in enumerate(M_ind)
            if length(M_ind) != 0 
                ind = M_ind[1]
                if ind[1] ∈ F_ind
                    index_position = findfirst(ind[1] .== F_ind)
                    start_val[i] = μ[index_position]
                end # latent means stay 0
            end
        end
    end

    return start_val
end

function is_in_Λ(ind_vec, F_ind)
    res = [!(ind[2] ∈ F_ind) & (ind[1] ∈ F_ind) for ind in ind_vec]
    return any(res)
end