using SEM, Symbolics, SparseArrays, Distributions, Optim, LineSearches

@variables x[1:13]

S = [x[1] 0 0 0 0 0 0 0
     0 x[2] 0 0 0 0 0 0
     0 0 x[3] 0 0 0 0 0
     0 0 0 x[4] 0 0 0 0
     0 0 0 0 x[5] 0 0 0
     0 0 0 0 0 x[6] 0 0
     0 0 0 0 0 0 x[7] x[9]
     0 0 0 0 0 0 x[9] x[8]]

F = [1.0 0 0 0 0 0 0 0
     0 1.0 0 0 0 0 0 0
     0 0 1.0 0 0 0 0 0
     0 0 0 1.0 0 0 0 0
     0 0 0 0 1.0 0 0 0
     0 0 0 0 0 1.0 0 0]

A = [0 0 0 0 0 0 1.0   0
     0 0 0 0 0 0 x[10] 0
     0 0 0 0 0 0 x[11] 0
     0 0 0 0 0 0 0     1.0
     0 0 0 0 0 0 0     x[12]
     0 0 0 0 0 0 0     x[13]
     0 0 0 0 0 0 0     0
     0 0 0 0 0 0 0     0]

S = sparse(S)
F = sparse(F)
A = sparse(A)

true_val = [repeat([1], 8)
            0.4
            repeat([0.8], 4)]

start_val = [repeat([1], 9)
             repeat([0.5], 4)]

imply_ml = RAMSymbolic(A, S, F, x, start_val)

imply_ml.Σ_function(imply_ml.Σ, true_val)

true_dist = MultivariateNormal(imply_ml.Σ)

x = transpose(rand(true_dist, 100000))
semobserved = SEM.SemObsCommon(data = x)


loss_ml = SemLoss((SEM.SemML(semobserved, [0.0], similar(start_val)), ))
diff = 
    SemDiffOptim(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8))

model_ml = SEM.Sem(semobserved, imply_ml, loss_ml, diff)
model_ml(true_val, 1.0, nothing, nothing)
solution = SEM.sem_fit(model_ml)

@test SEM.compare_estimates(true_val, solution.minimizer, .05)
