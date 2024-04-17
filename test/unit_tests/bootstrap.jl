solution_ml = sem_fit(model_ml)
bs = se_bootstrap(solution_ml; n_boot = 20)

update_se_hessian!(partable, solution_ml)
update_partable!(partable, solution_ml, bs, :se_boot)