using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff,
    Distributions, FiniteDiff

## Observed Data
three_path_dat = Feather.read("test/comparisons/three_path_dat.feather")
three_path_par = Feather.read("test/comparisons/three_path_par.feather")
three_path_start = Feather.read("test/comparisons/three_path_start.feather")
fitmeasures = Feather.read("test/comparisons/three_path_fitm.feather")

semobserved = SemObsCommon(data = Matrix(three_path_dat))

diff_fin = SemFiniteDiff(BFGS(), Optim.Options())

## Model definition
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

start_val = vcat(
    vec(var(Matrix(three_path_dat), dims = 1))./2,
    fill(0.05, 3),
    fill(0.0, 6),
    fill(1.0, 8),
    fill(0, 3)
    )

loss = Loss([SemML(semobserved, [0.0], similar(start_val))])

imply = ImplySymbolic(A, S, F, x, start_val)

model_fin = Sem(semobserved, imply, loss, diff_fin)

@benchmark solution_fin = sem_fit(model_fin)

solution_fin.minimum

par_order = [collect(21:34); collect(15:20); 2;3; 5;6;7; collect(9:14)]

all(
    abs.(solution_fin.minimizer .- three_path_par.est[par_order]
        ) .< 0.001*abs.(three_path_par.est[par_order]))

model_fin.observed.n_man

size(solution_fin.minimizer, 1)


# bollen
function sem_chi2(model, solution)
    F = solution.minimum
    S = model.observed.obs_cov
    N = model.observed.n_obs
    n = model.observed.n_man
    q = Float64(size(solution.minimizer, 1))
    df = 0.5*n*(n+1) - q

    if isnothing(model.imply.imp_mean)
        χ = (N - 1.0)*(F - logdet(S) - n)
    end

    d = Chisq(df)
    p = 1 - cdf(d, χ)

    return χ, p
end

using BenchmarkTools

value, p = sem_chi2(model_fin, solution_fin)

value ≈ fitmeasures.chisq[1]
p ≈ fitmeasures.pvalue[1]


function sem_fi(model, solution)
    F = solution.minimum
    S = model.observed.obs_cov
    N = model.observed.n_obs
    n = model.observed.n_man
    q = Float64(size(solution.minimizer, 1))

    df = 0.5*n*(n+1) - q
    if isnothing(model.imply.imp_mean)
        χ = (N-1.0)*(F - logdet(S) - n)
    end

    df₀ = 0.5*n*(n-1)
    Σ₀= Diagonal(S)
    χ₀ = (N-1.0)*(tr(inv(Σ₀)*S) + logdet(Σ₀) - logdet(S) - n)

    FI = (χ₀ - df₀ - χ + df)/(χ₀ - df₀)

    return FI

end

sem_fi(model_fin, solution_fin) ≈ fitmeasures.cfi[1]

function sem_RMSEA(model, solution)
    F = solution.minimum
    S = model.observed.obs_cov
    N = model.observed.n_obs
    n = model.observed.n_man
    q = Float64(size(solution.minimizer, 1))

    df = 0.5*n*(n+1) - q
    if isnothing(model.imply.imp_mean)
        χ = (N-1.0)*(F - logdet(S) - n)
    end

    R = (χ - df)/((N-1.0)*df)
    R < 0.0 ? R = 0.0 : R = R
    R = √(R)

    return R
end

sem_RMSEA(model_fin, solution_fin) ≈ fitmeasures.rmsea[1]


function sem_logl(model, solution)
    let F = solution.minimum, 
        N = model.observed.n_obs,
        n = model_fin.observed.n_man
            l = - N*0.5*(F + n*log(2pi))
    end
end

sem_logl(model_fin, solution_fin) ≈ fitmeasures.logl[1]

function logl(obs_mean, exp_cov, data_matr)
    exp_cov = Matrix(Hermitian(exp_cov))
    likelihood = -loglikelihood(MvNormal(obs_mean, exp_cov), transpose(data_matr))
    return likelihood
end

loglikelihood(MvNormal(vcat(Statistics.mean(Matrix(three_path_dat), dims = 1)...), model_fin.imply.imp_cov),transpose(model_fin.observed.data))

logl(vcat(Statistics.mean(Matrix(three_path_dat), dims = 1)...), model_fin.imply.imp_cov, model_fin.observed.data)

function sem_aic(model, solution)
    q = Float64(size(solution_fin.minimizer, 1))
    logl = sem_logl(model, solution)
    AIC = 2q - 2logl
    return AIC
end

function sem_aic_diff(model, solution)
    n = model.observed.n_man
    q = Float64(size(solution.minimizer, 1))
    df = 0.5*n*(n+1) - q

    χ = sem_chi2(model, solution)[1]
    AIC = χ - 2df
    return AIC
end

function sem_aic_bollen(model, solution)
    q = Float64(size(solution.minimizer, 1))
    χ = sem_chi2(model, solution)[1]
    AIC = χ + 2q
    return AIC
end

function sem_bic_andreas(model, solution)
    q = Float64(size(solution.minimizer, 1))
    χ = sem_chi2(model, solution)[1]
    N = model_fin.observed.n_obs

    BIC = χ + q*log(N)
    return BIC
end

function sem_bic(model, solution)
    q = Float64(size(solution_fin.minimizer, 1))
    logl = sem_logl(model, solution)

    BIC = q*log(N) - 2logl
    return BIC
end

# tests only work if lavaan meanstructure = FALSE

sem_aic(model_fin, solution_fin) ≈ fitmeasures.aic[1]

sem_bic(model_fin, solution_fin)

#

fitmeasures |> names


function f(func)
    getproperty(LinearAlgebra, Symbol(func))
end

f(inv)

getproperty(LinearAlgebra, Symbol("inv"))(rand(5,5))

v = inv

s = Symbol(v)

getproperty(LinearAlgebra, s)

pre = zeros(31,31)

FiniteDiff.finite_difference_hessian!(pre, model_fin, solution_fin.minimizer)

# hessian! from NLSolversBase is faster if the objective already was 
# TwiceDifferentiable, otherwise it is unclear
function sem_se(td::T, solution) where {T <: TwiceDifferentiable}
    H = hessian!(td, solution.minimizer)
    return H
end

function sem_se(model, solution)
    H = FiniteDiff.finite_difference_hessian(model, solution.minimizer)
    return H
end

sem_se(func, solution_fin) ≈ pre

using NLSolversBase

func = TwiceDifferentiable(model_fin, start_val)

obj = OnceDifferentiable(model_fin, start_val)

stored_state = Optim.initial_state(model_fin.diff.algorithm, model_fin.diff.options,
    obj, model_fin.imply.start_val)

result = optimize(
    obj,
    model_fin.imply.start_val,
    model_fin.diff.algorithm,
    model_fin.diff.options,
    stored_state)

@btime f()

diff = stored_state.invH - inv(pre)

maximum(diff)

se1 = .√diag(inv(pre))
se2 = .√diag(stored_state.invH)

diff = se1 - se2

maximum(diff)

z = solution_fin.minimizer / se1

all(
    abs.(se1 .- three_path_par.se[par_order]
        ) .< 0.001*abs.(three_path_par.se[par_order]))

all(
    abs.(z .- three_path_par.z[par_order]
        ) .< 0.001*abs.(three_path_par.z[par_order]))

