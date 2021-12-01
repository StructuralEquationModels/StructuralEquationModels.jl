using Pkg

Pkg.activate("test")

using CSV

Pkg.activate(".")

using DataFrames, SEM, Symbolics, 
    LinearAlgebra, SparseArrays, Optim, LineSearches,
    BenchmarkTools

cd("benchmark\\cfa")

include("functions.jl")

config = DataFrame(CSV.File("config.csv"))
config2 = copy(config)
config.backend .= "Optim.jl"
config2.backend .= "NLopt.jl"

config = [config; config2]

config = filter(row -> (row.Estimator == "ML") & (row.meanstructure == 0), config)

config = filter(
    row -> (row.Estimator == "ML") & 
    (row.n_factors == 5) & (row.n_items == 40) & (row.meanstructure == 0) & 
    (row.backend == "NLopt.jl"),
    config)

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

@benchmark sem_fit(models[1])

#ProfileView.@profview sem_fit(models[1])

function profile_fit(model, n)
    for i in 1:n
        sem_fit(model)
    end
end


ProfileView.@profview profile_fit(models[1], 100)

results = select(config, :Estimator, :n_factors, :n_items, :meanstructure, :backend)

results.mean_time_jl = mean.(getfield.(benchmarks, :times))/1000000000

CSV.write("results\\benchmarks_julia.csv", results, delim = ";")