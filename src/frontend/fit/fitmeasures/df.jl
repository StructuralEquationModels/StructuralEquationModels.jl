df(sem_fit) = df(sem_fit.model)

df(model::AbstractSemSingle) = n_dp(model) - n_par(model)

df(model::SemEnsemble) = n_dp(model) - n_par(model)

function n_dp(model::AbstractSemSingle)
    nman = n_man(model)
    ndp = 0.5(nman^2 + nman)
    if !isnothing(imply.μ)
        ndp += n_man(model)
    end
    return ndp
end

n_dp(model::SemEnsemble) = sum(n_dp.(model.sems))