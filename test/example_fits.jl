include("../test/example_models.jl");

one_fact_mod = model(one_fact_func, one_fact_dat, vcat(fill(1, 4), fill(0.5, 2)))
sem.fit(one_fact_mod)


@code_warntype one_fact_mod.objective(vcat(fill(1, 4), fill(0.5, 2)), one_fact_mod)

one_fact_reg_mod = model(
    one_fact_func,
    one_fact_dat,
    vcat(fill(1, 4),
    fill(0.5, 2));
    objective = sem.SemMLLasso(10, [true, true, true, true, false, false]))
fit(one_fact_mod)
