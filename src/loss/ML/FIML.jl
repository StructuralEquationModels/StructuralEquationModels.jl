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
    nobs_vars = nobserved_vars(pat)
    nmeas_vars = nmeasured_vars(pat)

    # generate linear indicies of co-observed variable pairs for each pattern
    Σ_linind = LinearIndices((nobs_vars, nobs_vars))
    pat_vars = findall(pat.measured_mask)
    ∇ind = vec(Σ_linind[pat_vars, pat_vars])

    return SemFIMLPattern(∇ind, zeros(nmeas_vars, nmeas_vars), Ref(NaN), zeros(nmeas_vars))
end

function prepare!(
    patloss::SemFIMLPattern,
    pat::SemObservedMissingPattern,
    Σ::AbstractMatrix,
    μ::AbstractVector,
)
    @inbounds @. @views begin
        patloss.Σ⁻¹ = Σ[pat.measured_mask, pat.measured_mask]
        patloss.μ_diff = pat.measured_mean - μ[pat.measured_mask]
    end
    Σ_chol = cholesky!(Symmetric(patloss.Σ⁻¹))
    patloss.logdet[] = logdet(Σ_chol)
    LinearAlgebra.inv!(Σ_chol) # updates loss.Σ⁻¹
    return patloss
end

prepare!(patloss::SemFIMLPattern, pat::SemObservedMissingPattern, implied::SemImplied) =
    prepare!(patloss, pat, implied.Σ, implied.μ)

function objective(patloss::SemFIMLPattern{T}, pat::SemObservedMissingPattern) where {T}
    F = patloss.logdet[] + dot(patloss.μ_diff, patloss.Σ⁻¹, patloss.μ_diff)
    if nsamples(pat) > 1
        F += dot(pat.measured_cov, patloss.Σ⁻¹)
        F *= nsamples(pat)
    end
    return F
end

function gradient!(JΣ, Jμ, patloss::SemFIMLPattern, pat::SemObservedMissingPattern)
    Σ⁻¹ = Symmetric(patloss.Σ⁻¹)
    μ_diff⨉Σ⁻¹ = patloss.μ_diff' * Σ⁻¹
    if nsamples(pat) > 1
        JΣ_pat = Σ⁻¹ * (I - pat.measured_cov * Σ⁻¹ - patloss.μ_diff * μ_diff⨉Σ⁻¹)
        JΣ_pat .*= nsamples(pat)
    else
        JΣ_pat = Σ⁻¹ * (I - patloss.μ_diff * μ_diff⨉Σ⁻¹)
    end
    @inbounds vec(JΣ)[patloss.∇ind] .+= vec(JΣ_pat)

    lmul!(2 * nsamples(pat), μ_diff⨉Σ⁻¹)
    @inbounds Jμ[pat.measured_mask] .+= μ_diff⨉Σ⁻¹'
    return nothing
end

"""
    SemFIML{T, W} <: SemLossFunction

Full information maximum likelihood (FIML) estimation.
Can handle observed data with missing values.

# Constructor

    SemFIML(observed::SemObservedMissing, implied::SemImplied)

# Arguments
- `observed::SemObservedMissing`: the observed part of the model
  (see [`SemObservedMissing`](@ref))
- `implied::SemImplied`: the implied part of the model
  (see [`SemImplied`](@ref))

# Examples
```julia
my_fiml = SemFIML(my_observed, my_implied)
```

# Interfaces
Analytic gradients are available.
"""
struct SemFIML{O, I, T, W} <: SemLoss{O, I}
    hessianeval::ExactHessian

    observed::O
    implied::I
    patterns::Vector{SemFIMLPattern{T}}

    imp_inv::Matrix{T}  # implied inverse

    commutator::CommutationMatrix

    interaction::W
end

############################################################################################
### Constructors
############################################################################################

function SemFIML(observed::SemObservedMissing, implied::SemImplied; kwargs...)
    if MeanStruct(implied) === NoMeanStruct
        """
        Full information maximum likelihood (FIML) can only be used with a meanstructure.
        Did you forget to set `Sem(..., meanstructure = true)`?
        """ |>
        ArgumentError |>
        throw
    end

    return SemFIML(
        ExactHessian(),
        observed,
        implied,
        [SemFIMLPattern(pat) for pat in observed.patterns],
        zeros(nobserved_vars(observed), nobserved_vars(observed)),
        CommutationMatrix(nvars(implied)),
        nothing,
    )
end

############################################################################################
### methods
############################################################################################

function evaluate!(objective, gradient, hessian, loss::SemFIML, params)
    isnothing(hessian) || error("Hessian not implemented for FIML")

    implied = SEM.implied(loss)
    observed = SEM.observed(loss)

    copyto!(loss.imp_inv, implied.Σ)
    Σ_chol = cholesky!(Symmetric(loss.imp_inv); check = false)

    if !isposdef(Σ_chol)
        isnothing(objective) || (objective = non_posdef_return(params))
        isnothing(gradient) || fill!(gradient, 1)
        return objective
    end

    @inbounds for (patloss, pat) in zip(loss.patterns, observed.patterns)
        prepare!(patloss, pat, implied)
    end

    scale = inv(nsamples(observed))
    isnothing(objective) || (objective = scale * F_FIML(eltype(params), loss))
    isnothing(gradient) || (∇F_FIML!(gradient, loss); gradient .*= scale)

    return objective
end
############################################################################################
### additional functions
############################################################################################

function ∇F_fiml_outer!(G, JΣ, Jμ, loss::SemFIML{O, I}) where {O, I <: SemImpliedSymbolic}
    mul!(G, loss.implied.∇Σ', JΣ) # should be transposed
    mul!(G, loss.implied.∇μ', Jμ, -1, 1)
end

function ∇F_fiml_outer!(G, JΣ, Jμ, loss::SemFIML)
    implied = loss.implied

    Iₙ = sparse(1.0I, size(implied.A)...)
    P = kron(implied.F⨉I_A⁻¹, implied.F⨉I_A⁻¹)
    Q = kron(implied.S * implied.I_A⁻¹', Iₙ)
    Q .+= loss.commutator * Q

    ∇Σ = P * (implied.∇S + Q * implied.∇A)

    ∇μ =
        implied.F⨉I_A⁻¹ * implied.∇M +
        kron((implied.I_A⁻¹ * implied.M)', implied.F⨉I_A⁻¹) * implied.∇A

    mul!(G, ∇Σ', JΣ) # actually transposed
    mul!(G, ∇μ', Jμ, -1, 1)
end

function F_FIML(::Type{T}, loss::SemFIML) where {T}
    F = zero(T)
    for (patloss, pat) in zip(loss.patterns, loss.observed.patterns)
        F += objective(patloss, pat)
    end
    return F
end

function ∇F_FIML!(G, loss::SemFIML)
    Jμ = zeros(nobserved_vars(loss))
    JΣ = zeros(nobserved_vars(loss)^2)

    for (patloss, pat) in zip(loss.patterns, loss.observed.patterns)
        gradient!(JΣ, Jμ, patloss, pat)
    end
    ∇F_fiml_outer!(G, JΣ, Jμ, loss)
end
