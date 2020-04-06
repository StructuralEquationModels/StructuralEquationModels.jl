include("../test/example_models.jl");

datas = (one_fact_dat, three_mean_dat, three_path_dat)
model_funcs = (one_fact_func, three_mean_func, three_path_func)
start_values = (
    vcat(fill(1, 4), fill(0.5, 2)),
    vcat(fill(1, 21), fill(0.5, 5)),
    vcat(fill(1, 20), fill(0.5, 6),
    mean(convert(Matrix{Float64}, three_mean_dat), dims = 1)[5:9])
    )

optimizers = (LBFGS(), GradientDescent(), Newton())


for i in 1:length(datas)
    for j in 1:length(optimizers)
        model = sem.model(model_funcs[i],
            datas[i],
            start_values[i])
    end
end


test = sem.model(model_funcs[1], datas[1], start_values[1])
Optim.minimizer(fit(test))

test = sem.model(model_funcs[2], datas[2], start_values[2])
Optim.minimizer(fit(test))

test.objective(start_values[1], test)
