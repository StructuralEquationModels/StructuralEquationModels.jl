# 

BIC(sem_fit::SemFit{Mi, S, Mo, O} where {Mi, S, Mo <: AbstractSemSingle, O}) = 
    BIC(
        sem_fit.minimum,
        sem_fit.solution,
        sem_fit.model,
        sem_fit.optimization_result,
        sem_fit.loss.functions...
        )

function BIC(
        minimum, 
        solution, 
        model::Sem{O, I, L, D} where {O, I <: RAM, L, D}, 
        optimization_result, 
        loss_ml::SemML)
    F_ML = minimum - logdet(model.observed.obs_cov) - p

    
end

function BIC(minimum, solution, model::Sem{O, I, L, D}, optimization_result, loss_ls::SemWLS)
    
end

function BIC(minimum, solution, model::Sem{O, I, L, D}, optimization_result, loss_ml::SemML, loss_ridge::SemRidge)
    
end

BIC(min, sol, mod, ores, l_ridge::SemRidge, l_ml::SemML) = BIC(min, sol, mod, ores, l_ml, l_ridge)