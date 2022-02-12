using DataFrames, StructuralEquationModels, Symbolics, 
    LinearAlgebra, SparseArrays, Optim, LineSearches,
    BenchmarkTools, CSV

date = string(ARGS...)

cd("cfa")

include("functions.jl")

config = DataFrame(CSV.File("config.csv"))
config2 = copy(config)
config.backend .= "Optim.jl"
config2.backend .= "NLopt.jl"

config = [config; config2]

config = filter(row -> (row.backend == "Optim.jl") & (row.meanstructure == 0), config)

data_vec = read_files("data", get_data_paths(config))
par_vec = read_files("parest", get_data_paths(config))
# start_vec = read_files("start", get_data_paths(config))

##############################################

models = gen_models(config, data_vec)

fits = get_fits(models)

correct = compare_estimates(fits, par_vec, config)

config.correct = correct

##############################################
# using MKL

benchmarks = benchmark_models(models)

results = select(config, :Estimator, :n_factors, :n_items, :meanstructure, :backend, :correct)

results.median_time_jl = median.(getfield.(benchmarks, :times))

CSV.write("results/benchmarks_julia_"*date*.csv", results, delim = ";")