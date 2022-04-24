using StructuralEquationModels, Distributions, Random, Optim, LineSearches
import StructuralEquationModels as SEM
include("helper.jl")

x = Symbol.("x".*string.(1:13))

S = [:x1 0 0 0 0 0 0 0
     0 :x2 0 0 0 0 0 0
     0 0 :x3 0 0 0 0 0
     0 0 0 :x4 0 0 0 0
     0 0 0 0 :x5 0 0 0
     0 0 0 0 0 :x6 0 0
     0 0 0 0 0 0 :x7 :x9
     0 0 0 0 0 0 :x9 :x8]

F = [1.0 0 0 0 0 0 0 0
     0 1.0 0 0 0 0 0 0
     0 0 1.0 0 0 0 0 0
     0 0 0 1.0 0 0 0 0
     0 0 0 0 1.0 0 0 0
     0 0 0 0 0 1.0 0 0]

A = [0 0 0 0 0 0 1.0   0
     0 0 0 0 0 0 :x10 0
     0 0 0 0 0 0 :x11 0
     0 0 0 0 0 0 0    1.0
     0 0 0 0 0 0 0    :x12
     0 0 0 0 0 0 0    :x13
     0 0 0 0 0 0 0    0
     0 0 0 0 0 0 0    0]

ram_matrices = RAMMatrices(;A = A, S = S, F = F, parameters = x, colnames = nothing)

true_val = [repeat([1], 8)
            0.4
            repeat([0.8], 4)]

start_val = [repeat([1], 9)
             repeat([0.5], 4)]

imply_ml = RAMSymbolic(;specification = ram_matrices, start_val = start_val)

imply_ml.Σ_function(imply_ml.Σ, true_val)

true_dist = MultivariateNormal(imply_ml.Σ)

Random.seed!(1234)
x = transpose(rand(true_dist, 100000))
semobserved = SemObsCommon(data = x)

loss_ml = SemLoss(SEM.SemML(;observed = semobserved, n_par = length(start_val)))

diff = 
    SemDiffOptim(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8))

model_ml = Sem(semobserved, imply_ml, loss_ml, diff)
model_ml(true_val, true, false, false)
solution_ml = sem_fit(model_ml)

@test isapprox(true_val, solution(solution_ml); atol = .05)