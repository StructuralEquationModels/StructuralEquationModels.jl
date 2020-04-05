# Maximum Likelihood Estimation

abstract type SemObjective end

struct SemML <: SemObjective end
function (objective::SemML)(parameters, model::model)
      obs_cov = model.obs.cov
      n_man = size(obs_cov, 1)
      Cov_Exp = imp_cov(parameters, model)
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) - log(det(obs_cov)) - n_man
      return F_ML
end

# SemMLMean doesnt work
struct SemMLMean <: SemObjective end
function (objective::SemMLMean)(parameters, model::model)
      obs_cov = model.obs_cov
      obs_mean = model.obs_mean
      n_man = size(obs_cov, 1)
      matrices = model.ram(parameters)
      Cov_Exp = imp_cov(model, parameters)
      Mean_Exp = matrices[2]*inv(I-matrices[3])*matrices[4]
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) +
                  transpose(obs_mean - Mean_Exp)*transpose(Cov_Exp)*
                        (obs_mean - Mean_Exp)
      return F_ML
end

### RegSem
struct SemMLLasso{P, W} <: SemObjective
    penalty::P
    wich::W
end
function (objective::SemMLLasso)(parameters, model::model)
      obs_cov = model.obs.cov
      obs_mean = model.obs.mean
      reg_vec = objective.which
      penalty = objective.penalty
      n_man = size(obs_cov, 1)
      matrices = model.ram(parameters)
      Cov_Exp = imp_cov(model, parameters)
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) -
                  log(det(obs_cov)) - n_man + penalty*sum(transpose(parameters)[reg_vec])
      return F_ML
end
# doesnt work
function ML_ridge(parameters; ram, obs_cov, reg_vec, penalty)
      obs_cov = model.obs_cov
      obs_mean = model.obs_mean
      reg_vec = model.rec_vec
      penalty = model.penalty
      n_man = size(obs_cov, 1)
      matrices = model.ram(parameters)
      Cov_Exp = matrices[2]*inv(I-matrices[3])*matrices[1]*transpose(inv(I-matrices[3]))*transpose(matrices[2])
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) -
                  log(det(obs_cov)) - n_man + penalty*sum(transpose(parameters)[reg_vec].^2)
      return F_ML
end
# FIML
### to add
