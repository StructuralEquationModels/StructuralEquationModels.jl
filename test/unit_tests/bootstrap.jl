solution_ml = sem_fit(model_ml)
bs = se_bootstrap(solution_ml; n_boot = 20)
se = se_hessian(solution_ml)

update_partable!(partable, solution_ml, se, :se)
update_partable!(partable, solution_ml, bs, :se_boot)