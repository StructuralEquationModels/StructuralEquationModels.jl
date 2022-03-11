#####################################################################################################
# Single Models
#####################################################################################################

# SemFit splices loss functions ---------------------------------------------------------------------
χ²(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    χ²(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

# RAM + SemML
χ²(sem_fit::SemFit, observed, imp::Union{RAM, RAMSymbolic}, diff, loss_ml::SemML) = 
    (nobs(sem_fit)-1)*(sem_fit.minimum - logdet(observed.obs_cov) - observed.n_man)

# FIML
function χ²(sem_fit::SemFit, observed::SemObsMissing, imp::RAM, diff, loss_ml::SemFIML)
    ll_H0 = minus2ll(sem_fit)
    ll_H1 = minus2ll(observed)
    chi2 = ll_H0 - ll_H1
    return chi2
end

#####################################################################################################
# Collections
#####################################################################################################