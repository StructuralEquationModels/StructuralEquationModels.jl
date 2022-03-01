# 

BIC(sem_fit::SemFit{}) = BIC(
    sem_fit.minimum,
    sem_fit.solution,
    sem_fit.model,
    sem_fit.optimization_result,
    )

function BIC(minimum, solution, model::Sem{O, I, L, D}, optimization_result)