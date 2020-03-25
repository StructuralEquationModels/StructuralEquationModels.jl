

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


function sem_obs_cov(model)
      push!(model, :obs_cov => cov(model[:data]))
end

function sem_imp_cov(model)
      push!(model, :imp_cov => imp_cov(model[:ram], model[:par]))
end

function sem_obs_mean(model)
      push!(model,
            :obs_mean =>
                  vec(mean(model[:data], dims = 1))::Vector{Float64}
            )
end


function sem_logl(model)
      push!(model, :logl =>
            logl(model[:obs_mean], model[:imp_cov], model[:data]))
end
