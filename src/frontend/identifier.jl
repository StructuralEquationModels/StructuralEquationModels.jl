#####################################################################################################
# get parameter identifier
#####################################################################################################

identifier(sem_fit::SemFit) = identifier(sem_fit.model)
identifier(model::AbstractSemSingle) = identifier(model.imply)
identifier(model::SemEnsemble) = model.identifier