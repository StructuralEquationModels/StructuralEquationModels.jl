n_man(sem_fit::SemFit) = n_man(sem_fit.model)

n_man(model::AbstractSemSingle) = n_man(model.observed)

"""
    n_man(sem_fit::SemFit)
    n_man(model::AbstractSemSingle)

Return the number of manifest variables.
"""
function n_man end