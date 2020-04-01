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

function imp_cov(parameters, model)
      ms = model.ram(parameters) # m(atrice)s
      invia = inv(I - ms[3]) # invers of I(dentitiy) minus A matrix
      cholesky(Symmetric(ms[2]*invia*ms[1]*transpose(invia)*transpose(ms[2])))
end
