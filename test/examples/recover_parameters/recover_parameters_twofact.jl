using StructuralEquationModels, Distributions, Random, Optim, LineSearches

include(
    joinpath(
        chop(dirname(pathof(StructuralEquationModels)), tail = 3),
        "test/examples/helper.jl",
    ),
)

x = Symbol.("x", 1:13)

S = [
    :x1 0 0 0 0 0 0 0
    0 :x2 0 0 0 0 0 0
    0 0 :x3 0 0 0 0 0
    0 0 0 :x4 0 0 0 0
    0 0 0 0 :x5 0 0 0
    0 0 0 0 0 :x6 0 0
    0 0 0 0 0 0 :x7 :x9
    0 0 0 0 0 0 :x9 :x8
]

F = [
    1.0 0 0 0 0 0 0 0
    0 1.0 0 0 0 0 0 0
    0 0 1.0 0 0 0 0 0
    0 0 0 1.0 0 0 0 0
    0 0 0 0 1.0 0 0 0
    0 0 0 0 0 1.0 0 0
]

A = [
    0 0 0 0 0 0 1.0   0
    0 0 0 0 0 0 :x10 0
    0 0 0 0 0 0 :x11 0
    0 0 0 0 0 0 0    1.0
    0 0 0 0 0 0 0    :x12
    0 0 0 0 0 0 0    :x13
    0 0 0 0 0 0 0    0
    0 0 0 0 0 0 0    0
]

ram_matrices = RAMMatrices(; A = A, S = S, F = F, param_labels = x, vars = nothing)

true_val = [
    repeat([1], 8)
    0.4
    repeat([0.8], 4)
]

start = [
    repeat([1], 9)
    repeat([0.5], 4)
]

implied_ml = RAMSymbolic(; specification = ram_matrices, start_val = start)

implied_ml.Σ_function(implied_ml.Σ, true_val)

true_dist = MultivariateNormal(implied_ml.Σ)

Random.seed!(1234)
x = transpose(rand(true_dist, 100_000))
semobserved = SemObservedData(data = x, specification = nothing)

loss_ml = SemLoss(SemML(; observed = semobserved, nparams = length(start)))

model_ml = Sem(semobserved, implied_ml, loss_ml)
objective!(model_ml, true_val)

optimizer = SemOptimizerOptim(
    BFGS(; linesearch = BackTracking(order = 3), alphaguess = InitialHagerZhang()),# m = 100),
    Optim.Options(; f_reltol = 1e-10, x_abstol = 1.5e-8),
)

solution_ml = fit(optimizer, model_ml)

@test true_val ≈ solution(solution_ml) atol = 0.05
