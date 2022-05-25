"""
    p(sem_fit::SemFit)

Return the p value computed from the χ² test statistic.
"""
p_value(sem_fit::SemFit) = 1 - cdf(Chisq(df(sem_fit)), χ²(sem_fit))

#####################################################################################################
# Single Models
#####################################################################################################

# SemFit splices loss functions ---------------------------------------------------------------------
#= p_value(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    p_value(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        ) =#

# RAM + SemML
#= p_value(sem_fit::SemFit, obs, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::Union{SemML, SemFIML, SemWLS}) = 
    1 - cdf(Chisq(df(sem_fit)), χ²(sem_fit)) =#

#####################################################################################################
# Collections
#####################################################################################################
#= 
p_value(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: SemEnsemble, O}) = 
    p_value(
        sem_fit,
        sem_fit.model,
        sem_fit.model.sems[1].loss.functions[1]
        )

p_value(sem_fit::SemFit, model::SemEnsemble, lossfun::Union{SemML, SemWLS}) = 1 - cdf(Chisq(df(sem_fit)), χ²(sem_fit)) =#