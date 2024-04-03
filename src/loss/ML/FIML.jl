############################################################################################
### Types
############################################################################################

# state of SemFIML for a specific missing pattern (`SemObservedMissingPattern` type)
struct SemFIMLPattern{T}
    ∇ind::Vector{Int}   # indices of co-observed variable pairs
    Σ⁻¹::Matrix{T}      # preallocated inverse of implied cov
    logdet::Ref{T}      # logdet of implied cov
    μ_diff::Vector{T}   # implied mean difference
end

# allocate arrays for pattern FIML
function SemFIMLPattern(pat::SemObservedMissingPattern)
    nobserved = nobserved_vars(pat)
    nmissed = nmissed_vars(pat)

    # linear indicies of co-observed variable pairs for each pattern
    Σ_linind = LinearIndices((nobserved, nobserved))
    ∇ind = vec([
        Σ_linind[CartesianIndex(x, y)] for x in findall(pat.obs_mask),
        y in findall(pat.obs_mask)
    ])

    return SemFIMLPattern(∇ind, zeros(nobserved, nobserved), Ref(NaN), zeros(nobserved))
end

function prepare!(fiml::SemFIMLPattern, pat::SemObservedMissingPattern, implied::SemImply)
    Σ = implied.Σ
    μ = implied.μ
    @inbounds @. @views begin
        fiml.Σ⁻¹ = Σ[pat.obs_mask, pat.obs_mask]
        fiml.μ_diff = pat.obs_mean - μ[pat.obs_mask]
    end
    Σ_chol = cholesky!(Symmetric(fiml.Σ⁻¹))
    fiml.logdet[] = logdet(Σ_chol)
    LinearAlgebra.inv!(Σ_chol) # updates fiml.Σ⁻¹
    #batch_sym_inv_update!(fiml, model)
    return fiml
end

function objective(fiml::SemFIMLPattern{T}, pat::SemObservedMissingPattern) where {T}
    F = fiml.logdet[] + dot(fiml.μ_diff, fiml.Σ⁻¹, fiml.μ_diff)
    if nsamples(pat) > 1
        F += dot(pat.obs_cov, fiml.Σ⁻¹)
        F *= nsamples(pat)
    end
    return F
end

function gradient!(JΣ, Jμ, fiml::SemFIMLPattern, pat::SemObservedMissingPattern)
    Σ⁻¹ = Symmetric(fiml.Σ⁻¹)
    μ_diff⨉Σ⁻¹ = fiml.μ_diff' * Σ⁻¹
    if n_obs(pat) > 1
        JΣ_pat = Σ⁻¹ * (I - pat.obs_cov * Σ⁻¹ - fiml.μ_diff * μ_diff⨉Σ⁻¹)
        JΣ_pat .*= nsamples(pat)
    else
        JΣ_pat = Σ⁻¹ * (I - fiml.μ_diff * μ_diff⨉Σ⁻¹)
    end
    @inbounds vec(JΣ)[fiml.∇ind] .+= vec(JΣ_pat)

    lmul!(2 * n_obs(pat), μ_diff⨉Σ⁻¹)
    @inbounds Jμ[pat.obs_mask] .+= μ_diff⨉Σ⁻¹'
    return nothing
end

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
struct SemFIML{T, W} <: SemLossFunction
    hessianeval::ExactHessian
    patterns::Vector{SemFIMLPattern{T}}

    imp_inv::Matrix{T}  # implied inverse

    commutator::CommutationMatrix

    interaction::W
end

############################################################################################
### Constructors
############################################################################################

function SemFIML(; observed::SemObservedMissing, specification, kwargs...)
    return SemFIML(
        ExactHessian(),
        [SemFIMLPattern(pat) for pat in observed.patterns],
        zeros(nobserved_vars(observed), nobserved_vars(observed)),
        CommutationMatrix(nvars(specification)),
        nothing,
    )
end

############################################################################################
### methods
############################################################################################

function evaluate!(
    objective,
    gradient,
    hessian,
    fiml::SemFIML,
    implied::SemImply,
    model::AbstractSemSingle,
    params,
)
    isnothing(hessian) || error("Hessian not implemented for FIML")

    if !check(fiml, model)
        isnothing(objective) || (objective = non_posdef_return(params))
        isnothing(gradient) || fill!(gradient, 1)
        return objective
    end

    prepare!(fiml, model)

    scale = inv(nsamples(observed(model)))
    isnothing(objective) ||
        (objective = scale * F_FIML(eltype(params), fiml, observed(model), model))
    isnothing(gradient) ||
        (∇F_FIML!(gradient, fiml, observed(model), model); gradient .*= scale)

    return objective
end

############################################################################################
### Recommended methods
############################################################################################

update_observed(lossfun::SemFIML, observed::SemObserved; kwargs...) =
    SemFIML(; observed = observed, kwargs...)

############################################################################################
### additional functions
############################################################################################

function ∇F_fiml_outer!(G, JΣ, Jμ, fiml::SemFIML, imply::SemImplySymbolic, model)
    mul!(G, imply.∇Σ', JΣ) # should be transposed
    mul!(G, imply.∇μ', Jμ, -1, 1)
end

function ∇F_fiml_outer!(G, JΣ, Jμ, fiml::SemFIML, imply, model)
    Iₙ = sparse(1.0I, size(imply.A)...)
    P = kron(imply.F⨉I_A⁻¹, imply.F⨉I_A⁻¹)
    Q = kron(imply.S * imply.I_A⁻¹', Iₙ)
    Q .+= fiml.commutator * Q

    ∇Σ = P * (imply.∇S + Q * imply.∇A)

    ∇μ = imply.F⨉I_A⁻¹ * imply.∇M + kron((imply.I_A⁻¹ * imply.M)', imply.F⨉I_A⁻¹) * imply.∇A

    mul!(G, ∇Σ', JΣ) # actually transposed
    mul!(G, ∇μ', Jμ, -1, 1)
end

function F_FIML(
    ::Type{T},
    fiml::SemFIML,
    observed::SemObservedMissing,
    model::AbstractSemSingle,
) where {T}
    F = zero(T)
    for (pat_fiml, pat) in zip(fiml.patterns, observed.patterns)
        F += objective(pat_fiml, pat)
    end
    return F
end

function ∇F_FIML!(G, fiml::SemFIML, observed::SemObservedMissing, model::AbstractSemSingle)
    Jμ = zeros(nobserved_vars(model))
    JΣ = zeros(nobserved_vars(model)^2)

    for (pat_fiml, pat) in zip(fiml.patterns, observed.patterns)
        gradient!(JΣ, Jμ, pat_fiml, pat)
    end
    ∇F_fiml_outer!(G, JΣ, Jμ, fiml, imply(model), model)
end

function prepare!(fiml::SemFIML, model::AbstractSemSingle)
    data = observed(model)::SemObservedMissing
    @inbounds for (pat_fiml, pat) in zip(fiml.patterns, data.patterns)
        prepare!(pat_fiml, pat, imply(model))
    end
    #batch_sym_inv_update!(fiml, model)
end

function check(fiml::SemFIML, model::AbstractSemSingle)
    copyto!(fiml.imp_inv, imply(model).Σ)
    a = cholesky!(Symmetric(fiml.imp_inv); check = false)
    return isposdef(a)
end
