using SEM, CSV, DataFrames, SparseArrays, Symbolics, LineSearches, Optim, Test, FiniteDiff
include("test_helpers.jl")

############################################################################
### observed data
############################################################################
using LinearAlgebra
dat = DataFrame(CSV.File("examples/data/data_dem.csv"))
par_ml = DataFrame(CSV.File("examples/data/par_dem_ml.csv"))
par_ls = DataFrame(CSV.File("examples/data/par_dem_ls.csv"))

dat = 
    select(
        dat,
        [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8])

        # observed
semobserved = SemObsCommon(data = Matrix{Float64}(dat))

############################################################################
### define models
############################################################################

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

start_val_fabin3 = start_fabin3(A, S, F, x, semobserved)
start_val_simple = start_simple(A, S, F, x)

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)


### start values
par_order = [collect(21:34); collect(15:20); 2;3; 5;6;7; collect(9:14)]
start_val_ml = Vector{Float64}(par_ml.start[par_order])
start_val_ls = Vector{Float64}(par_ls.start[par_order])
start_val_ridge = copy(start_val_ml)
start_val_ridge[16:20] .= .1

@test start_val_simple == [fill(1.0, 11); fill(0.05, 3); fill(0.0, 6); fill(0.5, 8); fill(0.0, 3)]
@test start_val_fabin3 ≈ start_val_ml

# start_val_snlls = Vector{Float64}(par_ls.start[par_order][21:31])

# loss
loss_ml = SemLoss((SemML(semobserved, length(start_val_ml)),))
loss_ls = SemLoss((SemWLS(semobserved, length(start_val_ml)),))
loss_ridge = SemLoss((SemML(semobserved, length(start_val_ml)), SemRidge(.001, 16:20, length(start_val_ml))))
#loss_ridge = SemLoss((SemML(semobserved, 1.0, similar(start_val_ml)), SemML(semobserved, 1.0, similar(start_val_ml))))

# loss_snlls = SemLoss([SemSWLS(semobserved, [0.0], similar(start_val_ml))])

# imply
imply_ml = RAMSymbolic(A, S, F, x, start_val_ml)
imply_ls = RAMSymbolic(A, S, F, x, start_val_ml; vech = true)
# imply_snlls = 

# diff
diff =
    SemDiffOptim(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100),
        #P = 0.5*inv(H0),
        #precondprep = (P, x) -> 0.5*inv(FiniteDiff.finite_difference_hessian(model_ls_ana, x))),
        Optim.Options(
            ;f_tol = 1e-10,
            x_tol = 1.5e-8))

# models
model_ml = Sem(semobserved, imply_ml, loss_ml, diff)
model_ls = Sem(semobserved, imply_ls, loss_ls, diff)
model_ridge = Sem(semobserved, imply_ml, loss_ridge, diff)


############################################################################
### test gradients
############################################################################


@testset "ml_gradients" begin
    @test test_gradient(model_ml, start_val_ml)
end

@testset "ls_gradients" begin
    @test test_gradient(model_ls, start_val_ls)
end

@testset "ridge_gradients" begin
    @test test_gradient(model_ridge, start_val_ml)
end

############################################################################
### test solution
############################################################################

@testset "ml_solution" begin
    solution_ml = sem_fit(model_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)
end

@testset "ls_solution" begin
    solution_ls = sem_fit(model_ls)
    @test SEM.compare_estimates(par_ls.est[par_order], solution_ls.minimizer, 0.01)
end
############################################################################
### test hessians
############################################################################

# loss
loss_ml = SemLoss((SemML(semobserved, length(start_val_ml)),))
loss_ls = SemLoss((SemWLS(semobserved, length(start_val_ml)),))

# imply
imply_ml = RAMSymbolic(A, S, F, x, start_val_ml; hessian = true)
imply_ls = RAMSymbolic(A, S, F, x, start_val_ml; vech = true, hessian = true)

# diff
diff = 
    SemDiffOptim(
        Newton(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100), 
        #P = 0.5*inv(H0),
        #precondprep = (P, x) -> 0.5*inv(FiniteDiff.finite_difference_hessian(model_ls_ana, x))), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8))

# models
model_ml = Sem(semobserved, imply_ml, loss_ml, diff)
model_ls = Sem(semobserved, imply_ls, loss_ls, diff)

@testset "ml_hessians" begin
    @test test_hessian(model_ml, start_val_ml)
end

@testset "ls_hessians" begin
    @test test_hessian(model_ls, start_val_ml)
end

@testset "ml_solution_hessian" begin
    solution_ml = sem_fit(model_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)
end

@testset "ls_solution_hessian" begin
    solution_ls = sem_fit(model_ls)
    @test SEM.compare_estimates(par_ls.est[par_order], solution_ls.minimizer, 0.01)
end

############################################################################
### starting values for standardized latents
############################################################################

par_ml_stdlv = DataFrame(CSV.File("examples/data/par_dem_ml_stdlv.csv"))

S2 =[x[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
    0     x[2]  0     0     0     0     0     0     0     0     0     0     0     0
    0     0     x[3]  0     0     0     0     0     0     0     0     0     0     0
    0     0     0     x[4]  0     0     0     x[12] 0     0     0     0     0     0
    0     0     0     0     x[5]  0     x[13] 0     x[14] 0     0     0     0     0
    0     0     0     0     0     x[6]  0     0     0     x[15] 0     0     0     0
    0     0     0     0     x[13] 0     x[7]  0     0     0     x[16] 0     0     0
    0     0     0     x[12] 0     0     0     x[8]  0     0     0     0     0     0
    0     0     0     0     x[14] 0     0     0     x[9]  0     x[17] 0     0     0
    0     0     0     0     0     x[15] 0     0     0     x[10] 0     0     0     0
    0     0     0     0     0     0     x[16] 0     x[17] 0     x[11] 0     0     0
    0     0     0     0     0     0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     0     0     0     0     0     1     0
    0     0     0     0     0     0     0     0     0     0     0     0     0     1]

F2 =[1.0 0 0 0 0 0 0 0 0 0 0 0 0 0
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

A2 =[0  0  0  0  0  0  0  0  0  0  0    x[18] 0     0
    0  0  0  0  0  0  0  0  0  0  0     x[19] 0     0
    0  0  0  0  0  0  0  0  0  0  0     x[20] 0     0
    0  0  0  0  0  0  0  0  0  0  0     0     x[21] 0
    0  0  0  0  0  0  0  0  0  0  0     0     x[22] 0
    0  0  0  0  0  0  0  0  0  0  0     0     x[23] 0
    0  0  0  0  0  0  0  0  0  0  0     0     x[24] 0
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[25]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[26]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[27]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[28]
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[29] 0     0
    0  0  0  0  0  0  0  0  0  0  0     x[30] x[31] 0]

start_val_fabin3 = start_fabin3(A2, S2, F2, x, semobserved)

par_order = [collect(21:31); collect(15:20); collect(1:14)]
@test start_val_fabin3 ≈ par_ml_stdlv.start[par_order]
par_order = [collect(21:34); collect(15:20); 2;3; 5;6;7; collect(9:14)]

############################################################################
### approximation of hessians
############################################################################

# loss
loss_ml = SemLoss((SemML(semobserved, length(start_val_ml); approx_H = true),))
loss_ls = SemLoss((SemWLS(semobserved, length(start_val_ml); approx_H = true),))

# imply
imply_ml = RAMSymbolic(A, S, F, x, start_val_ml)
imply_ls = RAMSymbolic(A, S, F, x, start_val_ml; vech = true)

# diff
diff = 
    SemDiffOptim(
        Newton(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100), 
        #P = 0.5*inv(H0),
        #precondprep = (P, x) -> 0.5*inv(FiniteDiff.finite_difference_hessian(model_ls_ana, x))), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8))

# models
model_ml = Sem(semobserved, imply_ml, loss_ml, diff)
model_ls = Sem(semobserved, imply_ls, loss_ls, diff)

@testset "ml_solution_apprhess" begin
    solution_ml = sem_fit(model_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)
end

@testset "ls_solution_apprhess" begin
    solution_ls = sem_fit(model_ls)
    @test SEM.compare_estimates(par_ls.est[par_order], solution_ls.minimizer, 0.01)
end
############################################################################
### meanstructure
############################################################################

par_ml = DataFrame(CSV.File("examples/data/par_dem_ml_mean.csv"))
par_ls = DataFrame(CSV.File("examples/data/par_dem_ls_mean.csv"))

# observed
semobserved = SemObsCommon(data = Matrix{Float64}(dat); meanstructure = true)

@Symbolics.variables x[1:38]

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

M = [x[32]; x[33]; x[34]; x[35]; x[36]; x[37]; x[38]; x[35]; x[36]; x[37]; x[38]; 0.0; 0.0; 0.0]

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)


### start values
par_order = [collect(29:42); collect(15:20); 2;3; 5;6;7; collect(9:14); collect(43:45); collect(21:24)]
start_val_ml = Vector{Float64}(par_ml.start[par_order])
start_val_ls = Vector{Float64}(par_ls.start[par_order])
# start_val_snlls = Vector{Float64}(par_ls.start[par_order][21:31])

# loss
loss_ml = SemLoss((SemML(semobserved, length(start_val_ml)),))
loss_ls = SemLoss((SemWLS(semobserved, length(start_val_ml); meanstructure = true),))
# loss_snlls = SemLoss([SemSWLS(semobserved, [0.0], similar(start_val_ml))])

# imply
imply_ml = RAMSymbolic(A, S, F, x, start_val_ml; M = M)
imply_ls = RAMSymbolic(A, S, F, x, start_val_ml; M = M, vech = true)
# imply_snlls = 

# diff
diff = 
    SemDiffOptim(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100), 
        #P = 0.5*inv(H0),
        #precondprep = (P, x) -> 0.5*inv(FiniteDiff.finite_difference_hessian(model_ls_ana, x))), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8))

# models
model_ml = Sem(semobserved, imply_ml, loss_ml, diff)
model_ls = Sem(semobserved, imply_ls, loss_ls, diff)

############################################################################
### test solution
############################################################################
@testset "ml_solution_meanstructure" begin
    solution_ml = sem_fit(model_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)
end

@testset "ls_solution_meanstructure" begin
    solution_ls = sem_fit(model_ls)
    @test SEM.compare_estimates(par_ls.est[par_order], solution_ls.minimizer, 0.01)
end

############################################################################
### test gradients
############################################################################

@testset "ml_gradients_meanstructure" begin
    @test test_gradient(model_ml, start_val_ml)
end

@testset "ls_gradients_meanstructure" begin
    @test test_gradient(model_ls, start_val_ls)
end

############################################################################
### fiml
############################################################################

############################################################################
### observed data
############################################################################

dat = DataFrame(CSV.read("examples/data/data_dem_fiml.csv", DataFrame; missingstring = "NA"))
par_ml = DataFrame(CSV.read("examples/data/par_dem_ml_fiml.csv", DataFrame))

dat = 
    select(
        dat,
        [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8])

# observed
semobserved = SemObsMissing(Matrix(dat))

############################################################################
### define models
############################################################################

@Symbolics.variables x[1:38]

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

M = [x[32]; x[33]; x[34]; x[35]; x[36]; x[37]; x[38]; x[35]; x[36]; x[37]; x[38]; 0.0; 0.0; 0.0]

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)

### start values
par_order = [collect(29:42); collect(15:20); 2;3; 5;6;7; collect(9:14); collect(43:45); collect(21:24)]
start_val_ml = Vector{Float64}(par_ml.start[par_order])

# loss
loss_ml = SemLoss((SEM.SemFIML(semobserved, length(start_val_ml)),))

# imply
imply_ml = RAMSymbolic(A, S, F, x, start_val_ml; M = M)

# diff
diff = 
    SemDiffOptim(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8))

diff = 
    SemDiffOptim(BFGS(), Optim.Options(;f_tol = 1e-10, x_tol = 1.5e-8))

    # models
model_ml = Sem(semobserved, imply_ml, loss_ml, diff)

############################################################################
### test gradients
############################################################################

using FiniteDiff

@testset "fiml_gradient" begin
    @test test_gradient(model_ml, start_val_ml)
end

############################################################################
### test solution
############################################################################

@testset "fiml_solution" begin
    solution_ml = sem_fit(model_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)
end