using Pkg

Pkg.activate("test")

using CSV

Pkg.activate(".")

using DataFrames, SEM, Symbolics, 
    LinearAlgebra, SparseArrays, Optim, LineSearches,
    BenchmarkTools

cd("benchmark\\sem")

include("functions.jl")

config = DataFrame(CSV.File("config.csv"))
config2 = copy(config)
config.backend .= "Optim.jl"
config2.backend .= "NLopt.jl"

config = [config; config2]

data_vec = read_files("data", get_data_paths(config))
# par_vec = read_files("parest", get_data_paths(config))
# start_vec = read_files("start", get_data_paths(config))

##############################################
models = gen_models(config, data_vec)

fits = get_fits(models)

##############################################

benchmarks = []
for model in models
    global modelxy = model
    bm = @benchmark sem_fit(modelxy)
    push!(benchmarks, bm)
end



results = select(config, :Estimator, :n_factors, :n_items, :meanstructure, :backend)

results.mean_time_jl = mean.(getfield.(benchmarks, :times))/1000000000

CSV.write("results\\benchmarks_julia.csv", results, delim = ";")