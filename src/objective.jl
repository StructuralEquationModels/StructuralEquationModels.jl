# Maximum Likelihood Estimation

struct SemML <: SemObjective end
function (objective::SemML)(parameters, model::model)
      obs_cov = model.obs.cov
      n_man = size(obs_cov, 1)
      matrices = model.ram(parameters)
      imp_cov = sem.imp_cov(matrices)
      F_ML = log(det(imp_cov)) + tr(obs_cov*inv(imp_cov)) - log(det(obs_cov)) - n_man
      if size(matrices, 1) == 4
          mean_diff = model.obs.mean - sem.imp_mean(matrices)
          F_ML = F_ML + transpose(mean_diff)*inv(imp_cov)*mean_diff
      end
      return F_ML
end

### RegSem
struct SemMLElastic{P, W} <: SemObjective
    ridge_penalty::P
    ridge_which::W
    lasso_penalty::P
    lasso_which::W
end

function (objective::SemMLElastic)(parameters, model::model)
      F_ML = SemML()(parameters, model)
      lasso = objective.lasso_penalty*sum(transpose(parameters)[objective.lasso_which])
      ridge = objective.ridge_penalty*sum(transpose(parameters)[objective.ridge_which].^2)
      F_ML + lasso + ridge
end

function SemMLLasso(which, penalty)
      ridge_penalty = zero(penalty)
      ridge_which = which
      SemMLElastic(ridge_penalty, lasso_penalty, which, penalty)
end

function SemMLRidge(which, penalty)
      lasso_penalty = zero(penalty)
      lasso_which = which
      SemMLElastic(penalty, penalty, lasso_which, lasso_penalty)
end
