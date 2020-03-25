# helper functions
function logl(obs_means, exp_cov, data_matr)
      exp_cov = Matrix(Hermitian(exp_cov))
      likelihood = -loglikelihood(MvNormal(obs_means, exp_cov), transpose(data_matr))
      return likelihood
end


function expected_cov(model, parameters)
      matrices = model(parameters)
      exp_cov =  matrices[2]*inv(I-matrices[3])*
      matrices[1]*transpose(inv(I-matrices[3]))*transpose(matrices[2])
      return exp_cov
end
