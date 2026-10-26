# Ordinary Maximum Likelihood Estimation

############################################################################################
### Types
############################################################################################
"""
Maximum likelihood estimation.

# Constructor

    SemML(observed, implied, refloss = nothing; approximate_hessian = false)

# Arguments
- `observed::SemObserved`: the observed part of the model
- `implied::SemImplied`: [`SemImplied`](@ref) instance
- `refloss::Union{SemML, Nothing}`: optional reference loss used to preserve
    loss-specific configuration and share the internal state when rebuilding
    a loss term, e.g. in [`replace_observed`](@ref)
- `approximate_hessian::Bool`: if hessian-based optimization is used, should the hessian be swapped for an approximation

# Examples
```julia
my_ml = SemML(my_observed, my_implied)
```

# Interfaces
Analytic gradients are available, and for models without a meanstructure
and `RAMSymbolic` implied type, also analytic hessians.
"""
struct SemML{O, I, HE <: HessianEval, M} <: SemLoss{O, I}
    observed::O
    implied::I
    hessianeval::HE

    # pre-allocated arrays to store intermediate results in evaluate!()
    obsXobs_1::M
    obsXobs_2::M
    obsXobs_3::M
    obsXvar_1::M
    varXvar_1::M
    varXvar_2::M
    varXvar_3::M
end

############################################################################################
### Constructors
############################################################################################

function SemML(
    observed::SemObserved,
    implied::SemImplied,
    refloss::Union{SemML, Nothing} = nothing;
    approximate_hessian::Bool = !isnothing(refloss) ?
                                HessianEval(refloss) === ApproxHessian : false,
    kwargs...,
)
    if observed isa SemObservedMissing
        @warn """
        ML estimation with `SemObservedMissing` will use an approximate covariance and mean estimated with EM algorithm.
        For more accurate results, consider using full information maximum likelihood (FIML) estimation or remove missing values in your data.
        A FIML model can be constructed with
        Sem(
            ...,
            observed = SemObservedMissing,
            loss = SemFIML,
            meanstructure = true
        )
        """
    end
    # check integrity
    check_observed_vars(observed, implied)
    @assert isnothing(refloss) || (
        nobserved_vars(refloss.observed) == nobserved_vars(observed) &&
        nvars(refloss.implied) == nvars(implied)
    )

    he = approximate_hessian ? ApproxHessian() : ExactHessian()
    obsXobs = parent(obs_cov(observed))
    nobs = nobserved_vars(observed)
    nvar = nvars(implied)

    return SemML{typeof(observed), typeof(implied), typeof(he), typeof(obsXobs)}(
        observed,
        implied,
        he,
        isnothing(refloss) ? similar(obsXobs) : refloss.obsXobs_1,
        isnothing(refloss) ? similar(obsXobs) : refloss.obsXobs_2,
        isnothing(refloss) ? similar(obsXobs) : refloss.obsXobs_3,
        isnothing(refloss) ? similar(obsXobs, (nobs, nvar)) : refloss.obsXvar_1,
        isnothing(refloss) ? similar(obsXobs, (nvar, nvar)) : refloss.varXvar_1,
        isnothing(refloss) ? similar(obsXobs, (nvar, nvar)) : refloss.varXvar_2,
        isnothing(refloss) ? similar(obsXobs, (nvar, nvar)) : refloss.varXvar_3,
    )
end

############################################################################################
### objective, gradient, hessian methods
############################################################################################

############################################################################################
### Symbolic Implied Types

function evaluate!(
    objective,
    gradient,
    hessian,
    loss::SemML{<:Any, <:SemImpliedSymbolic},
    par,
)
    implied = SEM.implied(loss)

    if !isnothing(hessian)
        (MeanStruct(implied) === HasMeanStruct) &&
            throw(DomainError(H, "hessian of ML + meanstructure is not available"))
    end

    Σ = implied.Σ
    Σₒ = obs_cov(observed(loss))

    Σ⁻¹ = copy!(loss.obsXobs_1, Σ)
    Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
    if !isposdef(Σ_chol)
        #@warn "∑⁻¹ is not positive definite"
        isnothing(objective) || (objective = non_posdef_objective(par))
        isnothing(gradient) || fill!(gradient, 1)
        isnothing(hessian) || copyto!(hessian, I)
        return objective
    end
    ld = logdet(Σ_chol)
    Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
    Σ⁻¹Σₒ = mul!(loss.obsXobs_2, Σ⁻¹, Σₒ)
    isnothing(objective) || (objective = ld + tr(Σ⁻¹Σₒ))

    if MeanStruct(implied) === HasMeanStruct
        μ = implied.μ
        μₒ = obs_mean(observed(loss))
        μ₋ = μₒ - μ

        isnothing(objective) || (objective += dot(μ₋, Σ⁻¹, μ₋))
        if !isnothing(gradient)
            ∇Σ = implied.∇Σ
            ∇μ = implied.∇μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            J = copyto!(loss.obsXobs_3, Σ⁻¹)
            mul!(J, Σ⁻¹Σₒ, Σ⁻¹, -1, 1)
            mul!(J, μ₋ᵀΣ⁻¹', μ₋ᵀΣ⁻¹, -1, 1)
            mul!(gradient, ∇Σ', vec(J))
            mul!(gradient, ∇μ', μ₋ᵀΣ⁻¹', -2, 1)
        end
    elseif !isnothing(gradient) || !isnothing(hessian)
        ∇Σ = implied.∇Σ
        Σ⁻¹ΣₒΣ⁻¹ = mul!(loss.obsXobs_3, Σ⁻¹Σₒ, Σ⁻¹)
        J = vec(Σ⁻¹ - Σ⁻¹ΣₒΣ⁻¹)'
        if !isnothing(gradient)
            mul!(gradient, ∇Σ', J')
        end
        if !isnothing(hessian)
            if HessianEval(loss) === ApproxHessian
                mul!(hessian, ∇Σ' * kron(Σ⁻¹, Σ⁻¹), ∇Σ, 2, 0)
            else
                ∇²Σ = implied.∇²Σ
                # inner
                implied.∇²Σ_eval!(∇²Σ, J, par)
                # outer
                H_outer = kron(2Σ⁻¹ΣₒΣ⁻¹ - Σ⁻¹, Σ⁻¹)
                mul!(hessian, ∇Σ' * H_outer, ∇Σ)
                hessian .+= ∇²Σ
            end
        end
    end
    return objective
end

############################################################################################
### Non-Symbolic Implied Types

function evaluate!(objective, gradient, hessian, loss::SemML, par)
    if !isnothing(hessian)
        error("hessian of ML + non-symbolic implied type is not available")
    end

    implied = SEM.implied(loss)

    Σ = implied.Σ
    Σₒ = obs_cov(observed(loss))

    Σ⁻¹ = copy!(loss.obsXobs_1, Σ)
    Σ_chol = cholesky!(Symmetric(Σ⁻¹); check = false)
    if !isposdef(Σ_chol)
        #@warn "Σ⁻¹ is not positive definite"
        isnothing(objective) || (objective = non_posdef_objective(par))
        isnothing(gradient) || fill!(gradient, 1)
        isnothing(hessian) || copyto!(hessian, I)
        return objective
    end
    ld = logdet(Σ_chol)
    Σ⁻¹ = LinearAlgebra.inv!(Σ_chol)
    Σ⁻¹Σₒ = mul!(loss.obsXobs_2, Σ⁻¹, Σₒ)

    if !isnothing(objective)
        objective = ld + tr(Σ⁻¹Σₒ)

        if MeanStruct(implied) === HasMeanStruct
            μ = implied.μ
            μₒ = obs_mean(observed(loss))
            μ₋ = μₒ - μ
            objective += dot(μ₋, Σ⁻¹, μ₋)
        end
    end

    if !isnothing(gradient)
        S = parent(implied.S)
        F⨉I_A⁻¹ = parent(implied.F⨉I_A⁻¹)
        I_A⁻¹ = parent(implied.I_A⁻¹)
        ∇A = implied.∇A
        ∇S = implied.∇S

        # reuse Σ⁻¹Σₒ to calculate I-Σ⁻¹Σₒ
        one_Σ⁻¹Σₒ = Σ⁻¹Σₒ
        lmul!(-1, one_Σ⁻¹Σₒ)
        one_Σ⁻¹Σₒ[diagind(one_Σ⁻¹Σₒ)] .+= 1

        C = mul!(
            loss.varXvar_1,
            F⨉I_A⁻¹',
            mul!(loss.obsXvar_1, mul!(loss.obsXobs_3, one_Σ⁻¹Σₒ, Σ⁻¹), F⨉I_A⁻¹),
        )
        mul!(
            gradient,
            ∇A',
            vec(mul!(loss.varXvar_3, C, mul!(loss.varXvar_2, S, I_A⁻¹'))),
            2,
            0,
        )
        mul!(gradient, ∇S', vec(C), 1, 1)

        if MeanStruct(implied) === HasMeanStruct
            μ = implied.μ
            μₒ = obs_mean(observed(loss))
            ∇M = implied.∇M
            M = implied.M
            μ₋ = μₒ - μ
            μ₋ᵀΣ⁻¹ = μ₋' * Σ⁻¹
            k = μ₋ᵀΣ⁻¹ * F⨉I_A⁻¹
            mul!(gradient, ∇M', k', -2, 1)
            mul!(
                gradient,
                ∇A',
                vec(mul!(loss.varXvar_1, k', (I_A⁻¹ * (M + S * k'))')),
                -2,
                1,
            )
            mul!(gradient, ∇S', vec(mul!(loss.varXvar_2, k', k)), -1, 1)
        end
    end

    return objective
end
