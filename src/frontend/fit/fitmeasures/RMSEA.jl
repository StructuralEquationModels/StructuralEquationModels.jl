"""
    RMSEA(fit::SemFit)

Return the RMSEA.
"""
function RMSEA end

RMSEA(fit::SemFit) = RMSEA(fit, fit.model)

function RMSEA(fit::SemFit, model::AbstractSemSingle)
    check_single_lossfun(model; throw_error = true)
    return RMSEA(dof(fit), χ²(fit), nsamples(fit)+rmsea_correction(model.loss.functions[1]))
end

function RMSEA(fit::SemFit, model::SemEnsemble)
    check_single_lossfun(model; throw_error = true)
    n = nsamples(fit)+model.n*rmsea_correction(model.sems[1].loss.functions[1])
    return sqrt(length(model.sems)) * RMSEA(dof(fit), χ²(fit), n)
end

function RMSEA(dof, chi2, N⁻)
    rmsea = (chi2 - dof) / (N⁻ * dof)
    rmsea = rmsea > 0 ? rmsea : 0
    return sqrt(rmsea)
end

# scaling corrections
rmsea_correction(::SemFIML) = 0
rmsea_correction(::SemML) = -1
rmsea_correction(::SemWLS) = -1
