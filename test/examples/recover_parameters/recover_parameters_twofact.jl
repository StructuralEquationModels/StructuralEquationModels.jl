using StructuralEquationModels, Distributions, Random, Optim, LineSearches

SEM = StructuralEquationModels

include(
    joinpath(
        chop(dirname(pathof(StructuralEquationModels)), tail = 3),
        "test/examples/helper.jl",
    ),
)

pars = Symbol.("x", 1:13)

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

ram_matrices = RAMMatrices(; A = A, S = S, F = F, param_labels = pars, vars = nothing)

true_val = [
    repeat([1], 8)
    0.4
    repeat([0.8], 4)
]

start = [
    repeat([1], 9)
    repeat([0.5], 4)
]

implied_sym = RAMSymbolic(ram_matrices)

implied_sym.Σ_eval!(implied_sym.Σ, true_val)

true_dist = MultivariateNormal(implied_sym.Σ)

Random.seed!(1234)
x = permutedims(rand(true_dist, 10^5), (2, 1))

observed = SemObservedData(data = x, specification = ram_matrices)

model_ml = Sem(SemML(observed, implied_sym))

objective!(model_ml, true_val)

optimizer = SemOptimizer(
    BFGS(; linesearch = BackTracking(order = 3), alphaguess = InitialHagerZhang()),# m = 100),
    Optim.Options(; f_reltol = 1e-10, x_abstol = 1.5e-8),
)

solution_ml = fit(optimizer, model_ml, start_val = start)

@test solution(solution_ml) ≈ true_val atol = 0.05
