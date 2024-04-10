############################################################################################
### Types
############################################################################################
"""
Full information maximum likelihood estimation. Can handle observed data with missings.

# Constructor

    SemFIML(;observed, specification, kwargs...)

# Arguments
- `observed::SemObservedMissing`: the observed part of the model
- `specification`: either a `RAMMatrices` or `ParameterTable` object

# Examples
```julia
my_fiml = SemFIML(observed = my_observed, specification = my_parameter_table)
```

# Interfaces
Analytic gradients are available.

# Extended help
## Implementation
Subtype of `SemLossFunction`.
"""
mutable struct SemFIML{INV, C, L, O, M, IM, I, T, U, W} <: SemLossFunction{ExactHessian}
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats

    ∇ind::O

    imp_mean::IM
    meandiff::M
    imp_inv::I

    mult::T

    commutation_indices::U

    interaction::W
end

############################################################################################
### Constructors
############################################################################################

function SemFIML(; observed::SemObservedMissing, specification, kwargs...)

    inverses = [zeros(nobserved_vars(pat), nobserved_vars(pat))
                for pat in observed.patterns]
    choleskys = Array{Cholesky{Float64,Array{Float64,2}},1}(undef, length(inverses))

    n_patterns = length(observed.patterns)
    logdets = zeros(n_patterns)

    imp_mean = [zeros(nobserved_vars(pat)) for pat in observed.patterns]
    meandiff = [zeros(nobserved_vars(pat)) for pat in observed.patterns]

    nman = n_man(observed)
    imp_inv = zeros(nman, nman)
    mult = similar.(inverses)

    # linear indicies of co-observed variable pairs for each pattern
    Σ_linind = LinearIndices((nman, nman))
    ∇ind = [[Σ_linind[CartesianIndex(x, y)] for x in findall(pat.obs_mask), y in findall(pat.obs_mask)]
            for pat in observed.patterns]

    commutation_indices = get_commutation_lookup(nvars(specification)^2)

    return SemFIML(
    inverses,
    choleskys,
    logdets,
    ∇ind,
    imp_mean,
    meandiff,
    imp_inv,
    mult,
    commutation_indices,
    nothing
    )
end

############################################################################################
### methods
############################################################################################

function evaluate!(objective, gradient, hessian,
                   semfiml::SemFIML, implied::SemImply, model::AbstractSemSingle, parameters)

    isnothing(hessian) || error("Hessian not implemented for FIML")

    if !check_fiml(semfiml, model)
        isnothing(objective) || (objective = non_posdef_return(parameters))
        isnothing(gradient) || fill!(gradient, 1)
        return objective
    end

    prepare_SemFIML!(semfiml, model)

    scale = inv(n_obs(observed(model)))
    isnothing(objective) || (objective = scale*F_FIML(observed(model), semfiml, model, parameters))
    isnothing(gradient) || (∇F_FIML!(gradient, observed(model), semfiml, model); gradient .*= scale)

    return objective
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(lossfun::SemFIML, observed::SemObserved; kwargs...) =
    SemFIML(;observed = observed, kwargs...)

############################################################################################
### additional functions
############################################################################################

function F_one_pattern(meandiff, inverse, obs_cov, logdet, N)
    F = logdet + dot(meandiff, inverse, meandiff)
    if N > one(N)
        F += dot(obs_cov, inverse)
    end
    return F*N
end

function ∇F_one_pattern(μ_diff, Σ⁻¹, S, obs_mask, ∇ind, N, Jμ, JΣ, model)
    diff⨉inv = μ_diff'*Σ⁻¹

    if N > one(N)
        JΣ[∇ind] .+= N*vec(Σ⁻¹*(I - S*Σ⁻¹ - μ_diff*diff⨉inv))
        @. Jμ[obs_mask] += (N*2*diff⨉inv)'

    else
        JΣ[∇ind] .+= vec(Σ⁻¹*(I - μ_diff*diff⨉inv))
        @. Jμ[obs_mask] += (2*diff⨉inv)'
    end

end

function ∇F_fiml_outer!(G, JΣ, Jμ, imply::SemImplySymbolic, model, semfiml)
    mul!(G, imply.∇Σ', JΣ) # should be transposed
    G .-= imply.∇μ' * Jμ
end

function ∇F_fiml_outer!(G, JΣ, Jμ, imply, model, semfiml)

    Iₙ = sparse(1.0I, size(imply.A)...)
    P = kron(imply.F⨉I_A⁻¹, imply.F⨉I_A⁻¹)
    Q = kron(imply.S*imply.I_A⁻¹', Iₙ)
    #commutation_matrix_pre_square_add!(Q, Q)
    Q2 = commutation_matrix_pre_square(Q, semfiml.commutation_indices)

    ∇Σ = P*(imply.∇S + (Q+Q2)*imply.∇A)

    ∇μ = imply.F⨉I_A⁻¹*imply.∇M + kron((imply.I_A⁻¹*imply.M)', imply.F⨉I_A⁻¹)*imply.∇A

    mul!(G, ∇Σ', JΣ) # actually transposed
    G .-= ∇μ' * Jμ
end

function F_FIML(observed::SemObservedMissing, semfiml, model, parameters)
    F = zero(eltype(parameters))
    for (i, pat) in enumerate(observed.patterns)
        F += F_one_pattern(
            semfiml.meandiff[i],
            semfiml.inverses[i],
            pat.obs_cov,
            semfiml.logdets[i],
            n_obs(pat))
    end
    return F
end

function ∇F_FIML!(G, observed::SemObservedMissing, semfiml, model)
    Jμ = zeros(n_man(model))
    JΣ = zeros(n_man(model)^2)

    for (i, pat) in enumerate(observed.patterns)
        ∇F_one_pattern(
            semfiml.meandiff[i],
            semfiml.inverses[i],
            pat.obs_cov,
            pat.obs_mask,
            semfiml.∇ind[i],
            n_obs(pat),
            Jμ,
            JΣ,
            model)
    end
    ∇F_fiml_outer!(G, JΣ, Jμ, imply(model), model, semfiml)
end

function prepare_SemFIML!(semfiml, model)
    copy_per_pattern!(semfiml, model)
    batch_cholesky!(semfiml, model)
    #batch_sym_inv_update!(semfiml, model)
    batch_inv!(semfiml, model)
    for (i, pat) in enumerate(observed(model).patterns)
        semfiml.meandiff[i] .= pat.obs_mean .- semfiml.imp_mean[i]
    end
end

function copy_per_pattern!(fiml::SemFIML, model::AbstractSem)
    Σ = imply(model).Σ
    μ = imply(model).μ
    data = observed(model)
    @inbounds @views for (i, pat) in enumerate(data.patterns)
        fiml.inverses[i] .= Σ[pat.obs_mask, pat.obs_mask]
        fiml.imp_means[i] .= μ[pat.obs_mask]
    end
end

function batch_cholesky!(semfiml, model)
    for i = 1:size(semfiml.inverses, 1)
        semfiml.choleskys[i] = cholesky!(Symmetric(semfiml.inverses[i]))
        semfiml.logdets[i] = logdet(semfiml.choleskys[i])
    end
    return true
end

function check_fiml(semfiml, model)
    copyto!(semfiml.imp_inv, imply(model).Σ)
    a = cholesky!(Symmetric(semfiml.imp_inv); check = false)
    return isposdef(a)
end
