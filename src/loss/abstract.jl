"""
    observed(loss::SemLoss) -> SemObserved

Returns the [*observed*](@ref SemObserved) part of a model.
"""
observed(loss::SemLoss) = loss.observed

"""
    implied(loss::SemLoss) -> SemImplied

Returns the [*implied*](@ref SemImplied) part of a model.
"""
implied(loss::SemLoss) = loss.implied

for f in (:nsamples, :obs_cov, :obs_mean)
    @eval $f(loss::SemLoss) = $f(observed(loss))
end

for f in (
    :vars,
    :nvars,
    :latent_vars,
    :nlatent_vars,
    :observed_vars,
    :nobserved_vars,
    :params,
    :nparams,
)
    @eval $f(loss::SemLoss) = $f(implied(loss))
end

function check_observed_vars(observed::SemObserved, implied::SemImplied)
    isnothing(observed_vars(implied)) ||
        observed_vars(observed) == observed_vars(implied) ||
        throw(
            ArgumentError(
                "Observed variables defined for \"observed\" and \"implied\" do not match.",
            ),
        )
end

check_observed_vars(sem::SemLoss) = check_observed_vars(observed(sem), implied(sem))

function check_observed_vars(loss::SemLoss, new_observed::SemObserved)
    observed_vars(new_observed) == observed_vars(SEM.observed(loss)) || throw(
        ArgumentError(
            "Observed variables of the loss term do not match the ones of the new observed data",
        ),
    )
end

############################################################################################
# replace_observed: SemLoss, AbstractLoss, LossTerm
############################################################################################

function replace_observed(loss::SemLoss, new_observed::SemObserved; kwargs...)
    check_observed_vars(loss, new_observed)
    # construct the new loss:
    # 1) replace the observed
    # 2) share the implied and its internal state with the original loss
    # 3) replicate the current loss configuration/share its internal state
    loss_ctor = typeof(loss).name.wrapper # get the loss constructor
    return loss_ctor(new_observed, SEM.implied(loss), loss)
end

function replace_observed(loss::SemLoss, data::Union{AbstractMatrix, DataFrame}; kwargs...)
    old_obs = SEM.observed(loss)
    obs_ctor = typeof(old_obs).name.wrapper
    new_observed = obs_ctor(data = data, observed_vars = observed_vars(old_obs))
    return replace_observed(loss, new_observed; kwargs...)
end

# non-SEM loss terms are unchanged
replace_observed(loss::AbstractLoss, ::Any; kwargs...) = loss

# LossTerm: delegate to inner loss
replace_observed(term::LossTerm, data; kwargs...) =
    LossTerm(replace_observed(loss(term), data; kwargs...), id(term), weight(term))

# returned objective if the implied Σ(par) matrix is not positive definite
function non_posdef_objective(par::AbstractVector)
    if eltype(par) <: AbstractFloat
        return floatmax(eltype(par))
    else
        return typemax(eltype(par))
    end
end
