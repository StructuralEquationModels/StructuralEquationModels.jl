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

############################################################################################
# replace_observed: SemLoss, AbstractLoss, LossTerm
############################################################################################

function replace_observed(loss::SemLoss, new_observed::SemObserved)
    old_obs = SEM.observed(loss)
    observed_vars(old_obs) == observed_vars(new_observed) || throw(
        ArgumentError(
            "observed_vars of the new data do not match the model: " *
            "expected $(observed_vars(old_obs)), got $(observed_vars(new_observed))",
        ),
    )
    return typeof(loss).name.wrapper(new_observed, SEM.implied(loss))
end

function replace_observed(loss::SemLoss, data::Union{AbstractMatrix, DataFrame})
    old_obs = SEM.observed(loss)
    new_observed =
        typeof(old_obs).name.wrapper(data = data, observed_vars = observed_vars(old_obs))
    return replace_observed(loss, new_observed)
end

# non-SEM loss terms are unchanged
replace_observed(loss::AbstractLoss, ::Any) = loss

# LossTerm: delegate to inner loss
replace_observed(term::LossTerm, data) =
    LossTerm(replace_observed(loss(term), data), id(term), weight(term))
