############################################################################
### based on -2ll
############################################################################

# SemFit splices loss functions ---------------------------------------------------------------------
BIC(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    BIC(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
BIC(sem_fit::SemFit, obs, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::SemML) =
    minus2ll(sem_fit) + log(obs.n_obs)*npar(imp)

function BIC(minimum, n_man, n_obs, n_par)
    BIC = minus2ll(minimum, n_obs, n_man) + log(n_obs)*n_par
    return BIC
end