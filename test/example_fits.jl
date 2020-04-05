include("../test/example_models.jl")
using Statistics, LinearAlgebra

one_fact_mod = model(one_fact_func, SemObs(one_fact_dat, mean = nothing), vcat(fill(1, 4), fill(0.5, 2)); optimizer = GradientDescent())
fit(one_fact_mod)
@code_warntype one_fact_mod.objective(vcat(fill(1, 4), fill(0.5, 2)), one_fact_mod)

one_fact_reg_mod = model(
one_fact_func,
one_fact_dat,
vcat(fill(1, 4),
fill(0.5, 2));
objective = SemMLLasso(10, [true, true, true, true, false, false]))
fit(one_fact_mod)
