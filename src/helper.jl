# helper functions
function logl(obs_mean, exp_cov, data_matr)
      exp_cov = Matrix(Hermitian(exp_cov))
      likelihood = -loglikelihood(MvNormal(obs_mean, exp_cov), transpose(data_matr))
      return likelihood
end

function logl_mean(exp_mean, exp_cov, data_matr)
      exp_cov = Matrix(Hermitian(exp_cov))
      likelihood = -loglikelihood(MvNormal(exp_mean, exp_cov), transpose(data_matr))
      return likelihood
end

function imp_cov(model, parameters)
      matrices = model(parameters)
      imp_cov =  matrices[2]*inv(I-matrices[3])*
      matrices[1]*transpose(inv(I-matrices[3]))*transpose(matrices[2])
      return imp_cov
end
