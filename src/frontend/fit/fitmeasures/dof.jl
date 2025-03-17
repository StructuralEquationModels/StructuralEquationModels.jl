"""
    dof(sem_fit::SemFit)
    dof(model::AbstractSem)

Return the degrees of freedom.
"""
function dof end

dof(sem_fit::SemFit) = dof(sem_fit.model)

dof(model::AbstractSem) = n_dp(model) - nparams(model)

function n_dp(model::AbstractSemSingle)
    nvars = nobserved_vars(model)
    ndp = 0.5(nvars^2 + nvars)
    if !isnothing(model.implied.μ)
        ndp += nvars
    end
    return ndp
end

n_dp(model::SemEnsemble) = sum(n_dp.(model.sems))
