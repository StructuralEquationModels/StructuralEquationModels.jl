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
function delta_method!(model)
      par = model[:par]
      ram = model[:ram]
      obs_mean = model[:obs_mean]
      data = model[:data]

      if model[:opt] == "LBFGS"
            fun = param -> logl(obs_mean,
                              imp_cov(ram, param),
                              data)
            se =
            sqrt.(diag(inv(ForwardDiff.hessian(fun, par))))
      elseif model[:opt] == "Newton"
            fun = TwiceDifferentiable(param -> logl(model[:obs_mean],
                                          imp_cov(model[:ram], param),
                                          model[:data]),
                                          model[:par],
                                          autodiff = :forward)
            se = sqrt.(diag(inv(hessian!(fun, par))))
      else
            error("Your Optimizer is not supported")
      end
      z = par./se
      p = cdf.(Normal(), -abs(z))
      push!(model, :se => se, :p => p, :z => z)
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
