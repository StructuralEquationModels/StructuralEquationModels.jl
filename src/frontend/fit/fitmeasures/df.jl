"""
    df(sem_fit::SemFit)
    df(model::AbstractSem)

Return the degrees of freedom.
"""
function df end

df(sem_fit::SemFit) = df(sem_fit.model)

df(model::AbstractSem) = n_dp(model) - n_par(model)

function n_dp(model::AbstractSemSingle)
    nman = n_man(model)
    ndp = 0.5(nman^2 + nman)
    if !isnothing(model.imply.μ)
        ndp += n_man(model)
    end
    return ndp
end

n_dp(model::SemEnsemble) = sum(n_dp.(model.sems))