# wrapper to call the optimizer
function optim_sem(model, obs_cov, start, est = ML, optim = "LBFGS")
      if optim == "LBFGS"
            objective = parameters -> est(parameters, model, obs_cov)
            result = optimize(objective, start, LBFGS(), autodiff = :forward)
      elseif optim == "Newton"
            objective = TwiceDifferentiable(
                  parameters -> est(parameters, model, obs_cov),
                  start,
                  autodiff = :forward)
            result = optimize(objective, start)
      else
            error("Unknown Optimizer")
      end
      #result = optimize(objective, start, LBFGS(), autodiff = :forward)
      return result
end

# function to fit SEM and obtain a big fitted object
function fit_sem(model, data, start, est = ML, optim = "LBFGS")
      data_matr = convert(Matrix{Float64}, data)
      obs_cov = cov(data_matr)
      obs_means::Vector{Float64} = vec(mean(data_matr, dims = 1))

      # fit model
      result = optim_sem(model, obs_cov, start, est, optim)
      # obtain variables to compute logl
      parameters = Optim.minimizer(result)
      exp_cov = expected_cov(model, parameters)

      fitted_model = Dict{Symbol, Any}(
      :parameters => parameters,
      :data => data_matr,
      :obs_cov => obs_cov,
      :exp_cov => expected_cov(model, parameters),
      :obs_means => obs_means,
      :model => model,
      :logl => logl(obs_means, exp_cov, data_matr),
      :opt_result => result,
      :optimizer => optim,
      :start => start
      )
      return fitted_model
end

# compute standard errors and p-values
function delta_method(fit)
      parameters = fit[:parameters]
      model = fit[:model]
      obs_means = fit[:obs_means]
      data = fit[:data]

      if fit[:optimizer] == "LBFGS"
            fun = param -> logl(obs_means,
                              expected_cov(model, param),
                              data)
            se =
            sqrt.(diag(inv(ForwardDiff.hessian(fun, parameters))))
      elseif fit[:optimizer] == "Newton"
            fun = TwiceDifferentiable(param -> logl(fit[:obs_means],
                                          expected_cov(fit[:model], param),
                                          fit[:data]),
                                          fit[:start],
                                          autodiff = :forward)
            se = sqrt.(diag(inv(hessian!(fun, parameters))))
      else
            error("Your Optimizer is not supported")
      end
      z = parameters./se
      p = pdf(Normal(), z)
      return DataFrame(se = se,
                        z = z,
                        p = p)
end



# function to use in decision trees. Returns only loglik instead of
# a big fitted object. Should be modified to return the fit-measure of
# interest
function fit_in_tree(model, data, start, est = ML, optim = "LBFGS")
      data_matr = convert(Array{Float64}, data)
      obs_cov = cov(data_matr)
      obs_means = vec(mean(data_matr, dims = 1))

      # fit model
      result = optim_sem(model, obs_cov, start, est, optim)
      # obtain variables to compute logl
      parameters = Optim.minimizer(result)
      exp_cov = expected_cov(model, parameters)

      # compute logl
      likelihood = logl(obs_means, exp_cov, data_matr)
      return likelihood
end
