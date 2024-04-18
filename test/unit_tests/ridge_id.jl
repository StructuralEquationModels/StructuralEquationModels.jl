model_ridge_id = Sem(
    specification = spec,
    data = dat,
    loss = (SemML, SemRidge),
    α_ridge = 0.001,
    which_ridge = [:x16, :x17, :x18, :x19, :x20],
    optimizer = semoptimizer,
)
