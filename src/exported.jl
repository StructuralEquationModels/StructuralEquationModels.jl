# function to fit SEM and obtain a big fitted object
function sem_fit!(model)
      prepare_model!(model)
      # fit model
      result = opt_sem(model)
      # obtain variables to compute logl
      push!(model, :par => Optim.minimizer(result))
      push!(model, :opt_result => result)
      sem_imp_cov!(model)
end

function prepare_model!(model)
      if ismissing(model[:obs_cov])
            sem_obs_cov!(model)
      end
      if ismissing(model[:obs_mean])
            sem_obs_mean!(model)
      end
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



function sem_obs_cov!(model)
      push!(model, :obs_cov => cov(model[:data]))
end

function sem_imp_cov!(model)
      push!(model, :imp_cov => imp_cov(model[:ram], model[:par]))
end

function sem_obs_mean!(model)
      push!(model,
            :obs_mean =>
                  vec(mean(model[:data], dims = 1))::Vector{Float64}
            )
end


function sem_logl!(model)
      push!(model, :logl =>
            logl(model[:obs_mean], model[:imp_cov], model[:data]))
end

function sem_est!(model, est)
      push!(model, :est => est)
end

function sem_opt!(model, opt)
      push!(model, :opt => opt)
end
