using DataFrames, StructuralEquationModels, Symbolics, 
    LinearAlgebra, SparseArrays, Optim, LineSearches,
    BenchmarkTools, CSV

cd("benchmark/cfa")

include("functions.jl")

config = DataFrame(CSV.File("config.csv"))
config2 = copy(config)
config.backend .= "Optim.jl"
config2.backend .= "NLopt.jl"

config = [config; config2]

config = filter(row -> (row.Estimator == "ML") & (row.meanstructure == 0), config)

data_vec = read_files("data", get_data_paths(config))
par_vec = read_files("parest", get_data_paths(config))
# start_vec = read_files("start", get_data_paths(config))

##############################################
models = gen_models(config, data_vec)

fits = get_fits(models)

##############################################

benchmarks = benchmark_models(models)

results = select(config, :Estimator, :n_factors, :n_items, :meanstructure, :backend)

results.mean_time_jl = mean.(getfield.(benchmarks, :times))

CSV.write("results/benchmarks_julia.csv", results, delim = ";")