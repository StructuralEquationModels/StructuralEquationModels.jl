# function to fit SEM and obtain a big fitted object
function sem_fit!(model)
      prepare_model!(model)
      # fit model
      result = opt_sem(model)
      # store obtained solution in model
      model.par = Optim.minimizer(result)
      model.opt_result = result
      #update implied covariance
      sem_imp_cov!(model)
end

function prepare_model!(model)
      if isnothing(model.obs_cov)
            sem_obs_cov!(model)
      end
      if isnothing(model.obs_mean)
            sem_obs_mean!(model)
      end
      if model.mstruc
            sem_est!(model, ML_mean)
      elseif !model.mstruc
            sem_est!(model, ML)
      end
      if isnothing(model.reg)
      elseif model.reg == "lasso"
            sem_est!(model, ML_lasso)
      elseif model.reg == "ridge"
            sem_est!(model, ML_ridge)
      end
end

# compute standard errors and p-values
function delta_method!(model)
      par = model.par
      ram = model.ram
      obs_mean = model.obs_mean
      data = model.data

      if model.opt == "LBFGS"
            fun = parameter -> logl(obs_mean,
                              imp_cov(ram, parameter),
                              data)
            se =
            sqrt.(diag(inv(ForwardDiff.hessian(fun, par))))
      elseif model.opt == "Newton"
            fun = TwiceDifferentiable(parameter -> logl(obs_mean,
                                          imp_cov(ram, parameter),
                                          data),
                                          par,
                                          autodiff = :forward)
            se = sqrt.(diag(inv(hessian!(fun, par))))
      else
            error("Your Optimizer is not supported")
      end
      z = par./se
      p = cdf.(Normal(), -abs.(z))
      push!(model, :se => se, :p => p, :z => z)
end

function sem_replace!(x, new_x)
    setindex!(x, new_x,
        1:size(x, 1),
        1:size(x, 2))
end

function sem_copy_replace!(x, new_x)
    setindex!(x, new_x,
        1:size(x, 1),
        1:size(x, 2))
end

function sem_obs_cov!(model)
      model.obs_cov = Distributions.cov(model.data)
end

function sem_imp_cov!(model)
      model.imp_cov = imp_cov(model.ram, model.par)
end

function sem_obs_mean!(model)
      model.obs_mean = mean(model.data, dims = 1)
end


function sem_logl!(model)
      model.logl = logl(
                        model.obs_mean,
                        model.imp_cov,
                        model.data)
end

function sem_est!(model, est)
      model.est = est
end

function sem_opt!(model, opt)
      model.opt = opt
end
