using SEM, CSV, DataFrames, SparseArrays, Symbolics, LineSearches, Optim, Test

## Observed Data
dat = DataFrame(CSV.File("examples/data/data_dem.csv"))
par_ml = DataFrame(CSV.File("examples/data/par_dem_ml.csv"))
par_ls = DataFrame(CSV.File("examples/data/par_dem_ls.csv"))

dat = 
    select(
        dat,
        [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8])

        # observed
semobserved = SemObsCommon(data = Matrix{Float64}(dat))

## Model definition
@Symbolics.variables x[1:31]

#x = rand(31)

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

start_val_ml = Vector{Float64}(par_ml.start[par_order])
# start_val_ls = Vector{Float64}(par_ls.start[par_order])
# start_val_snlls = Vector{Float64}(par_ls.start[par_order][21:31])

# loss
loss_ml = SemLoss((SemML(semobserved, [0.0], similar(start_val_ml)),))
# loss_ls = SemLoss([SemWLS(semobserved, [0.0], similar(start_val_ml))])
# loss_snlls = SemLoss([SemSWLS(semobserved, [0.0], similar(start_val_ml))])

# imply
imply_ml = RAMSymbolic(A, S, F, x, start_val_ml)
# imply_ls = 
# imply_snlls = 

diff = 
    SemDiffOptim(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100), 
        #P = 0.5*inv(H0),
        #precondprep = (P, x) -> 0.5*inv(FiniteDiff.finite_difference_hessian(model_ls_ana, x))), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8))


model_ml = Sem(semobserved, imply_ml, loss_ml, diff)

model_ml(start_val_ml, 1.0, nothing, nothing)

using FiniteDiff

@testset "ml_gradients" begin
    grad = similar(start_val_ml)
    grad .= 0.0
    model_ml(start_val_ml, 1.0, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ml(x, 1.0, nothing, nothing), start_val_ml)
    grad .= 0.0
    model_ml(start_val_ml, nothing, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ml(x, 1.0, nothing, nothing), start_val_ml)
end

# fit
solution_ml = sem_fit(model_ml)
@test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)