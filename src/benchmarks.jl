using sem, Test, Feather, BenchmarkTools, Distributions,
        Optim

function holz_onef_mod(x)
    S = [x[1] 0 0 0
        0 x[2] 0 0
        0 0 x[3] 0
        0 0 0 x[4]]

    F = [1 0 0 0
        0 1 0 0
        0 0 1 0]

    A = [0 0 0 1
        0 0 0 x[5]
        0 0 0 x[6]
        0 0 0 0]

    return (S, F, A)
end

function holz_onef_mod_mean(x)
    S = [x[1] 0 0 0
        0 x[2] 0 0
        0 0 x[3] 0
        0 0 0 x[4]]

    F = [1 0 0 0
        0 1 0 0
        0 0 1 0]

    A = [0 0 0 1
        0 0 0 x[5]
        0 0 0 x[6]
        0 0 0 0]

    M = [0
        x[8]
        x[9]
        x[7]]

    return (S, F, A, M)
end

holz_onef_dat = Feather.read("test/comparisons/holz_onef_dat.feather")
holz_onef_par = Feather.read("test/comparisons/holz_onef_par.feather")

mymod_lbfgs = model(holz_onef_mod, holz_onef_dat,
            [0.5, 0.5, 0.5, 0.5, 1.0, 1.0])

sem_fit!(mymod_lbfgs)

@benchmark begin
    mymod_lbfgs =
        model(holz_onef_mod, holz_onef_dat,
        [0.5, 0.5, 0.5, 0.5, 1.0, 1.0])
    sem_fit!(mymod_lbfgs)
end

@benchmark begin
    mymod_lbfgs =
    model(holz_onef_mod, holz_onef_dat,
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
    opt = "test")
    sem_fit!(mymod_lbfgs)
end

@benchmark begin
    mymod_lbfgs =
    model(holz_onef_mod, holz_onef_dat,
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
    obs_cov = Distributions.cov(convert(Matrix{Float64}, holz_onef_dat)),
    obs_mean = mean(convert(Matrix{Float64}, holz_onef_dat), dims = 1),
    est = sem.ML)
    objective = parameters ->
            sem.ML_test(parameters, mymod_lbfgs.ram, mymod_lbfgs.obs_cov)
    result =
            optimize(objective, mymod_lbfgs.par, LBFGS(),
            autodiff = :forward)
end

mymod_newton =
    model(holz_onef_mod, holz_onef_dat,
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
    opt = "Newton")
sem_fit!(mymod_newton)
