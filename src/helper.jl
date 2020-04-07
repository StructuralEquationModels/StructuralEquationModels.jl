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

function imp_cov(matrices)
      ms = matrices
      invia = inv(I - ms[3]) # invers of I(dentity) minus A matrix
      imp = ms[2]*invia*ms[1]*transpose(invia)*transpose(ms[2])
      return imp
end

function imp_mean(matrices)
      ms = matrices
      ms[2]*inv(I-ms[3])*ms[4]
end
