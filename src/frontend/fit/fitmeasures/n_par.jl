############################################################################
### get number of parameters
############################################################################

n_par(fit::SemFit) = n_par(fit.model)

n_par(model::AbstractSemSingle) = n_par(model.imply)

n_par(model::SemEnsemble) = n_par(model.identifier)

n_par(identifier::dict) = length(keys(identifier))

# SemFit splices loss functions ---------------------------------------------------------------------
#= n_par(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    n_par(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        ) =#

# RAM + SemML
#= n_par(sem_fit::SemFit, obs, imply::Union{RAM, RAMSymbolic}, diff, loss) = n_par(imply)

n_par(model::AbstractSemSingle) = n_par(model.imply) =#
