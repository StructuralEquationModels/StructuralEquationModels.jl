# wrapper to call the optimizer
function opt_sem(model)
      if model[:opt] == "LBFGS"
            objective = parameters ->
                  model[:est](parameters, model[:ram], model[:obs_cov])
            result =
                  optimize(objective, model[:par], LBFGS(), autodiff = :forward)
      elseif model[:opt] == "Newton"
            objective = TwiceDifferentiable(
                  parameters -> model[:est](parameters, model[:ram], model[:obs_cov]),
                  model[:par],
                  autodiff = :forward)
            result = optimize(objective, model[:par])
      else
            error("Unknown Optimizer")
      end
      #result = optimize(objective, start, LBFGS(), autodiff = :forward)
      return result
end
