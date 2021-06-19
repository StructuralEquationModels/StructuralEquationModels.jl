cd("C:\\Users\\maxim\\.julia\\dev\\sem\\test\\comparisons\\factor_simulation")

include("factor_functions.jl")

config = DataFrame(CSV.File("config_factor.csv"))

data_vec = read_files("data", get_data_paths(config))
par_vec = read_files("parest", get_data_paths(config))
start_vec = read_files("start", get_data_paths(config))

#############################################
models = gen_models(config, data_vec, start_vec)

benchmarks = benchmark_models(models)

fits = get_fits(models)
##############################################

all(
    abs.(solution_fin.minimizer .- par.est[par_order]
        ) .< 0.05*abs.(par.est[par_order]))