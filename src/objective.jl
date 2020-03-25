# Maximum Likelihood Estimation
function ML(parameters, model, obs_cov)
      matrices = model(parameters)
      Cov_Exp =  matrices[2]*inv(I-matrices[3])*matrices[1]*transpose(inv(I-matrices[3]))*transpose(matrices[2])
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) - log(det(obs_cov)) - 3
      return F_ML
end

# FIML
### to add
