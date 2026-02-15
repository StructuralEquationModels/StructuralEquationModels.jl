"""
    RMSEA(fit::SemFit)

Return the RMSEA.
"""
function RMSEA end

RMSEA(fit::SemFit) = RMSEA(fit, fit.model)

function RMSEA(fit::SemFit, model::AbstractSemSingle)
    check_uniform_lossfun(model)
    return RMSEA(dof(fit), χ²(fit), nsamples(fit)-dof_correction(model.loss.functions[1]))
end

function RMSEA(fit::SemFit, model::SemEnsemble)
    check_single_lossfun(model; throw_error = true)
    n = nsamples(fit)-model.n*dof_correction(model.sems[1].loss.functions[1])
    return sqrt(length(model.sems)) * RMSEA(dof(fit), χ²(fit), n)
end

function RMSEA(dof, chi2, c)
    rmsea = (chi2 - dof) / (c * dof)
    rmsea = rmsea > 0 ? rmsea : 0
    return sqrt(rmsea)
end

