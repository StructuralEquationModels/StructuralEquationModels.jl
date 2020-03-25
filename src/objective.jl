# Maximum Likelihood Estimation
function ML(parameters, ram, obs_cov)
      n_man = size(obs_cov, 1)
      matrices = ram(parameters)
      Cov_Exp =  matrices[2]*inv(I-matrices[3])*matrices[1]*transpose(inv(I-matrices[3]))*transpose(matrices[2])
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) - log(det(obs_cov)) - n_man
      return F_ML
end

# FIML
### to add
