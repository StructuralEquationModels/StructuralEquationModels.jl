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

function imp_cov(ram::ram)
      invia = inv(I - ram.A) # invers of I(dentity) minus A matrix
      imp = ram.F*invia*ram.S*transpose(invia)*transpose(ram.F)
      return imp
end

function imp_mean(matrices)
      ms = matrices
      ms[2]*inv(I-ms[3])*ms[4]
end
