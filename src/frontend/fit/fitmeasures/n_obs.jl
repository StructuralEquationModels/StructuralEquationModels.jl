"""
    n_obs(sem_fit::SemFit)
    n_obs(model::AbstractSemSingle)
    n_obs(model::SemEnsemble)

Return the number of observed data points.

For ensemble models, return the sum over all submodels.
"""
function n_obs end

n_obs(sem_fit::SemFit) = n_obs(sem_fit.model)

n_obs(model::AbstractSemSingle) = n_obs(model.observed)

n_obs(model::SemEnsemble) = sum(n_obs, model.sems)