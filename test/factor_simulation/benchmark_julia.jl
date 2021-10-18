cd("C:\\Users\\maxim\\.julia\\dev\\sem\\test\\comparisons\\factor_simulation")

include("factor_functions.jl")

config = DataFrame(CSV.File("config_factor.csv"))

data_vec = read_files("data", get_data_paths(config))
par_vec = read_files("parest", get_data_paths(config))
start_vec = read_files("start", get_data_paths(config))

##############################################
models = gen_models(config, data_vec, start_vec)

par_order = [model[2] for model in models]

models = [model[1] for model in models]

fits = get_fits(models)
##############################################

check_solution(fits, par_vec, par_order)

##############################################

benchmarks = benchmark_models(models)

##############################################

start_val = models[1].imply.start_val
grad = similar(start_val)
grad2 = similar(start_val)

@btime models[1](start_val, grad)

FiniteDiff.finite_difference_gradient!(grad2, models[2], start_val)

isapprox(grad, grad2)

diff_ana = 
    SemAnalyticDiff(
        :LD_LBFGS, 
        nothing,
        (grad_fiml,))


using Cthulhu

start_val = convert(Vector{Float64}, start_vec[1].est[par_order[1]])

@descend models[1](start_val)

grad = similar(start_val)

models[1](start_val, grad)

using FiniteDiff

grad2 = FiniteDiff.finite_difference_gradient(models[1], start_val)

grad ≈ grad2

@descend models[1](start_val, grad)


### NLopt

sol = sem.sem_fit_nlopt(models[1])

maximum(abs.(sol[2] .- par_vec[1].est[par_order[1]]))

using BenchmarkTools

@benchmark sem.sem_fit_nlopt(models[1])

sol2 = sem_fit_nlopt(models[2])

@time sol2 = sem_fit_nlopt(models[2])

maximum(abs.(sol2[2] .- par_vec[2].est[par_order[2]]))