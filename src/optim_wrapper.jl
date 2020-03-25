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
