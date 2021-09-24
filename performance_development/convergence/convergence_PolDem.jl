using sem, Arrow, ModelingToolkit, 
    LinearAlgebra, SparseArrays, DataFrames, 
    Optim, LineSearches, StatsBase, Distributions, Statistics, Random,
    Distances

Random.seed!(2848102)

################################################### read data ###################################################

cd("test")

## Observed Data
data = DataFrame(Arrow.Table("comparisons/data_dem.arrow"))
par_ml = DataFrame(Arrow.Table("comparisons/par_dem_ml.arrow"))
par_ls = DataFrame(Arrow.Table("comparisons/par_dem_ls.arrow"))

data = 
    select(
        data, 
        [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8])

data = convert(Matrix{Float64}, data)

n = 100
n_obs = size(data, 1)
n_obs_gen = 75
max_iter = 200

################################################### draw bootstrap samples ###################################################

function bootstrap_samples(data, nobs, n_obs_gen, n)
    obs = 1:nobs
    rowind = [sample(obs, n_obs_gen) for i in 1:n]
    samples = [data[rowind[i], :] for i in 1:n]
    return samples
end

data_boot = bootstrap_samples(data, n_obs, n_obs_gen, n)

################################################### draw normal samples from observed distribution ###################################################

distribution_observed = MvNormal(vec(mean(data; dims = 1)), cov(data))
data_obs = [rand(distribution_observed, n_obs_gen) for i in 1:n]
data_obs = permutedims.(data_obs)

################################################### draw normal samples from implied distribution ###################################################
# observed
function get_imp_cov(x)
    S =[x[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
    0     x[2]  0     0     0     0     0     0     0     0     0     0     0     0
    0     0     x[3]  0     0     0     0     0     0     0     0     0     0     0
    0     0     0     x[4]  0     0     0     x[15] 0     0     0     0     0     0
    0     0     0     0     x[5]  0     x[16] 0     x[17] 0     0     0     0     0
    0     0     0     0     0     x[6]  0     0     0     x[18] 0     0     0     0
    0     0     0     0     x[16] 0     x[7]  0     0     0     x[19] 0     0     0
    0     0     0     x[15] 0     0     0     x[8]  0     0     0     0     0     0
    0     0     0     0     x[17] 0     0     0     x[9]  0     x[20] 0     0     0
    0     0     0     0     0     x[18] 0     0     0     x[10] 0     0     0     0
    0     0     0     0     0     0     x[19] 0     x[20] 0     x[11] 0     0     0
    0     0     0     0     0     0     0     0     0     0     0     x[12] 0     0
    0     0     0     0     0     0     0     0     0     0     0     0     x[13] 0
    0     0     0     0     0     0     0     0     0     0     0     0     0     x[14]]

    F =[1.0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 1 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 1 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 1 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 1 0 0 0]

    A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
        0  0  0  0  0  0  0  0  0  0  0     x[21] 0     0
        0  0  0  0  0  0  0  0  0  0  0     x[22] 0     0
        0  0  0  0  0  0  0  0  0  0  0     0     1     0
        0  0  0  0  0  0  0  0  0  0  0     0     x[23] 0
        0  0  0  0  0  0  0  0  0  0  0     0     x[24] 0
        0  0  0  0  0  0  0  0  0  0  0     0     x[25] 0
        0  0  0  0  0  0  0  0  0  0  0     0     0     1
        0  0  0  0  0  0  0  0  0  0  0     0     0     x[26]
        0  0  0  0  0  0  0  0  0  0  0     0     0     x[27]
        0  0  0  0  0  0  0  0  0  0  0     0     0     x[28]
        0  0  0  0  0  0  0  0  0  0  0     0     0     0
        0  0  0  0  0  0  0  0  0  0  0     x[29] 0     0
        0  0  0  0  0  0  0  0  0  0  0     x[30] x[31] 0]
    invA = inv(I-A)
    imp_cov = F*invA*S*invA'*F'
    return imp_cov
end

par_order = [collect(21:34); collect(15:20); 2;3; 5;6;7; collect(9:14)]
imp_cov = get_imp_cov(convert(Vector{Float64}, par_ml.est[par_order]))


distribution_implied = MvNormal(vec(mean(data; dims = 1)), Symmetric(imp_cov))
data_imp = [rand(distribution_implied, n_obs_gen) for i in 1:n]
data_imp = permutedims.(data_imp)

############################## specify the models #########################
@ModelingToolkit.variables x[1:31]

S =[x[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
    0     x[2]  0     0     0     0     0     0     0     0     0     0     0     0
    0     0     x[3]  0     0     0     0     0     0     0     0     0     0     0
    0     0     0     x[4]  0     0     0     x[15] 0     0     0     0     0     0
    0     0     0     0     x[5]  0     x[16] 0     x[17] 0     0     0     0     0
    0     0     0     0     0     x[6]  0     0     0     x[18] 0     0     0     0
    0     0     0     0     x[16] 0     x[7]  0     0     0     x[19] 0     0     0
    0     0     0     x[15] 0     0     0     x[8]  0     0     0     0     0     0
    0     0     0     0     x[17] 0     0     0     x[9]  0     x[20] 0     0     0
    0     0     0     0     0     x[18] 0     0     0     x[10] 0     0     0     0
    0     0     0     0     0     0     x[19] 0     x[20] 0     x[11] 0     0     0
    0     0     0     0     0     0     0     0     0     0     0     x[12] 0     0
    0     0     0     0     0     0     0     0     0     0     0     0     x[13] 0
    0     0     0     0     0     0     0     0     0     0     0     0     0     x[14]]

F =[1.0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 0 0]

A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[21] 0     0
    0  0  0  0  0  0  0  0  0  0  0     x[22] 0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0  0  0     0     x[23] 0
    0  0  0  0  0  0  0  0  0  0  0     0     x[24] 0
    0  0  0  0  0  0  0  0  0  0  0     0     x[25] 0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[26]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[27]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[28]
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[29] 0     0
    0  0  0  0  0  0  0  0  0  0  0     x[30] x[31] 0]

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)
    
par_order = [collect(21:34); collect(15:20); 2;3; 5;6;7; collect(9:14)]

start_val_ml = [fill(1.0, 14); fill(0, 6); fill(1, 8); fill(0, 3)]
start_val_ls = [fill(1.0, 14); fill(0, 6); fill(1, 8); fill(0, 3)]
start_val_snlls = [fill(1.0, 8); fill(0, 3)]

function gen_model(data, opt_algo, start_val, A, S, F, lossfun, implyfun, parsubset; opt_options = nothing)
    # observed
    semobserved = SemObsCommon(data = data)
    # diff
    diff = SemFiniteDiff(opt_algo, opt_options)
    # loss
    loss = Loss([lossfun(semobserved, [0.0], similar(start_val_ml))])
    # imply
    imply = implyfun(A, S, F, x[parsubset], start_val)
    # model
    model = Sem(semobserved, imply, loss, diff)
    return model
end

function check_approx(estimated, true_par; tol = 0.05)
    all(abs.(estimated .- true_par) .< tol*abs.(true))
end

function swap_observed(model, data)
    new_model = Sem(SemObsCommon(data = data), deepcopy(model.imply), deepcopy(model.loss), deepcopy(model.diff))
    return new_model
end

function com_dist_max(a, b)
    dist = maximum(abs.(a-b))
    return dist
end

function get_omega(mod, fit)
    omega = inv(mod.imply.G'*mod.loss.functions[1].V*mod.imply.G)*mod.imply.G'*mod.loss.functions[1].sᵀV'
    omega = omega[[1, 2, 3, 4, 6, 9, 11, 13, 14, 16, 17, 18, 19, 20, 5, 7, 8, 10, 12, 15]]
    return vcat(omega, fit.minimizer)
end

function analyze_fits(fits, models; true_pars = nothing, postcond = false)
    converged = Optim.converged.(fits)
    minimizer = Optim.minimizer.(fits)
    iterations = Optim.iterations.(fits)
    time = getfield.(fits, :time_run)
    if postcond
        minimizer = get_omega.(models, fits)
    end
    res = DataFrame(
        converged = converged, 
        minimizer = minimizer,
        iterations = iterations,
        time = time)
    if !isnothing(true_pars)
        dist_euk = euclidean.(minimizer, [true_pars])
        res.dist_euk = dist_euk
        dist_max = com_dist_max.(minimizer, [true_pars])
        res.dist_max = dist_max
    end
    return res
end

function summarize_fits(fit_table)
    away = fit_table.dist_euk .> 20
    fit_table.converged[away] .= false
    n_converged = sum(fit_table.converged)
    mean_iter = mean(fit_table.iterations[fit_table.converged])
    mean_dist = mean(fit_table.dist_euk[fit_table.converged])
    mean_time = mean(fit_table.time[fit_table.converged])
    return DataFrame(n_converged = n_converged, mean_iter = mean_iter, mean_dist = mean_dist, mean_time = mean_time)
end
########################################################## simulation implied ############################################################

#= mod = gen_model(data, Newton(), start_val_snlls, A, S, F, sem.SemSWLS, sem.ImplySymbolicSWLS, 21:31; opt_options = Optim.Options())
fit = sem_fit(mod)
check_approx(get_omega(mod, fit), par_ls.est[par_order]) =#

models_imp_ls = [gen_model(data_imp[i], Newton(), start_val_ls, A, S, F, sem.SemWLS, sem.ImplySymbolicWLS, 1:31; opt_options = Optim.Options(;iterations = max_iter)) for i in 1:n]
fits_imp_ls = sem_fit.(models_imp_ls)
res_imp_ls = analyze_fits(fits_imp_ls, models_imp_ls; true_pars = par_ml.est[par_order])
summarize_fits(res_imp_ls)

models_imp_ml = [gen_model(data_imp[i], BFGS(), start_val_ml, A, S, F, SemML, ImplySymbolic, 1:31; opt_options = Optim.Options(;iterations = max_iter)) for i in 1:n]
fits_imp_ml = sem_fit.(models_imp_ml)
res_imp_ml = analyze_fits(fits_imp_ml, models_imp_ml; true_pars = par_ml.est[par_order])
summarize_fits(res_imp_ml)

models_imp_snlls = [gen_model(data_imp[i], Newton(), start_val_snlls, A, S, F, sem.SemSWLS, sem.ImplySymbolicSWLS, 21:31; opt_options = Optim.Options(;iterations = max_iter)) for i in 1:n]
fits_imp_snlls = sem_fit.(models_imp_snlls)
res_imp_snlls = analyze_fits(fits_imp_snlls, models_imp_snlls; true_pars = par_ml.est[par_order], postcond = true)
summarize_fits(res_imp_snlls)

########################################################## simulation bootstrap ############################################################
models_imp_ls = [gen_model(data_boot[i], BFGS(), start_val_ls, A, S, F, sem.SemWLS, sem.ImplySymbolicWLS, 1:31; opt_options = Optim.Options(;iterations = max_iter)) for i in 1:n]
fits_imp_ls = sem_fit.(models_imp_ls)
res_imp_ls = analyze_fits(fits_imp_ls, models_imp_ls; true_pars = par_ml.est[par_order])
summarize_fits(res_imp_ls)

models_imp_ml = [gen_model(data_boot[i], BFGS(), start_val_ml, A, S, F, SemML, ImplySymbolic, 1:31; opt_options = Optim.Options(;iterations = max_iter)) for i in 1:n]
fits_imp_ml = sem_fit.(models_imp_ml)
res_imp_ml = analyze_fits(fits_imp_ml, models_imp_ml; true_pars = par_ml.est[par_order])
summarize_fits(res_imp_ml)

models_imp_snlls = [gen_model(data_boot[i], Newton(), start_val_snlls, A, S, F, sem.SemSWLS, sem.ImplySymbolicSWLS, 21:31; opt_options = Optim.Options(;iterations = max_iter)) for i in 1:n]
fits_imp_snlls = sem_fit.(models_imp_snlls)
res_imp_snlls = analyze_fits(fits_imp_snlls, models_imp_snlls; true_pars = par_ml.est[par_order], postcond = true)
summarize_fits(res_imp_snlls)

########################################################## simulation obs ############################################################
models_imp_ls = [gen_model(data_obs[i], BFGS(), start_val_ls, A, S, F, sem.SemWLS, sem.ImplySymbolicWLS, 1:31; opt_options = Optim.Options(;iterations = max_iter)) for i in 1:n]
fits_imp_ls = sem_fit.(models_imp_ls)
res_imp_ls = analyze_fits(fits_imp_ls, models_imp_ls; true_pars = par_ml.est[par_order])
summarize_fits(res_imp_ls)

models_imp_ml = [gen_model(data_obs[i], BFGS(), start_val_ml, A, S, F, SemML, ImplySymbolic, 1:31; opt_options = Optim.Options(;iterations = max_iter)) for i in 1:n]
fits_imp_ml = sem_fit.(models_imp_ml)
res_imp_ml = analyze_fits(fits_imp_ml, models_imp_ml; true_pars = par_ml.est[par_order])
summarize_fits(res_imp_ml)

models_imp_snlls = [gen_model(data_obs[i], BFGS(), start_val_snlls, A, S, F, sem.SemSWLS, sem.ImplySymbolicSWLS, 21:31; opt_options = Optim.Options(;iterations = max_iter)) for i in 1:n]
fits_imp_snlls = sem_fit.(models_imp_snlls)
res_imp_snlls = analyze_fits(fits_imp_snlls, models_imp_snlls; true_pars = par_ml.est[par_order], postcond = true)
summarize_fits(res_imp_snlls)
