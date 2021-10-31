using SEM, CSV, DataFrames, SparseArrays, Symbolics, LineSearches, Optim, Test

############################################################################
### observed data
############################################################################

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

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)


### start values
par_order = [collect(21:34); collect(15:20); 2;3; 5;6;7; collect(9:14)]
start_val_ml = Vector{Float64}(par_ml.start[par_order])
start_val_ls = Vector{Float64}(par_ls.start[par_order])
# start_val_snlls = Vector{Float64}(par_ls.start[par_order][21:31])

# loss
loss_ml = SemLoss((SemML(semobserved, 1.0, similar(start_val_ml)),))
loss_ls = SemLoss((SemWLS(semobserved),))
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

############################################################################
### test gradients
############################################################################

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

@testset "ls_gradients" begin
    grad = similar(start_val_ls)
    grad .= 0.0
    model_ls(start_val_ls, 1.0, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ls(x, 1.0, nothing, nothing), start_val_ls)
    grad .= 0.0
    model_ls(start_val_ls, nothing, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ls(x, 1.0, nothing, nothing), start_val_ls)
end

############################################################################
### test solution
############################################################################

solution_ml = sem_fit(model_ml)
@test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)

solution_ls = sem_fit(model_ls)
@test SEM.compare_estimates(par_ls.est[par_order], solution_ls.minimizer, 0.01)

############################################################################
### test hessians
############################################################################

# loss
loss_ml = SemLoss((SemML(semobserved, 1.0, similar(start_val_ml)),))
loss_ls = SemLoss((SemWLS(semobserved),))

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
    hessian = zeros(size(start_val_ml, 1), size(start_val_ml, 1))
    grad = zeros(size(start_val_ml, 1))
    
    model_ml(start_val_ml, 1.0, nothing, hessian)
    @test hessian ≈ FiniteDiff.finite_difference_hessian(x -> model_ml(x, 1.0, nothing, nothing), start_val_ml) rtol = 1/1000
    
    hessian .= 0.0

    model_ml(start_val_ml, nothing, grad, hessian)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ml(x, 1.0, nothing, nothing), start_val_ml)
    @test hessian ≈ FiniteDiff.finite_difference_hessian(x -> model_ml(x, 1.0, nothing, nothing), start_val_ml) rtol = 1/1000
end

@testset "ls_hessians" begin
    hessian = zeros(size(start_val_ml, 1), size(start_val_ml, 1))
    grad = zeros(size(start_val_ml, 1))
    
    model_ls(start_val_ml, 1.0, nothing, hessian)
    @test hessian ≈ FiniteDiff.finite_difference_hessian(x -> model_ls(x, 1.0, nothing, nothing), start_val_ml) rtol = 1/1000
    
    hessian .= 0.0

    model_ls(start_val_ml, nothing, grad, hessian)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ls(x, 1.0, nothing, nothing), start_val_ml)
    @test hessian ≈ FiniteDiff.finite_difference_hessian(x -> model_ls(x, 1.0, nothing, nothing), start_val_ml) rtol = 1/1000
end

solution_ml = sem_fit(model_ml)
@test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)

solution_ls = sem_fit(model_ls)
@test SEM.compare_estimates(par_ls.est[par_order], solution_ls.minimizer, 0.01)

############################################################################
### approximation of hessians
############################################################################

# loss
loss_ml = SemLoss((SemML(semobserved, 1.0, similar(start_val_ml); approx_H = true),))
loss_ls = SemLoss((SemWLS(semobserved; approx_H = true),))

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

solution_ml = sem_fit(model_ml)
@test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)

solution_ls = sem_fit(model_ls)
@test SEM.compare_estimates(par_ls.est[par_order], solution_ls.minimizer, 0.01)

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
loss_ml = SemLoss((SemML(semobserved, 1.0, similar(start_val_ml)),))
loss_ls = SemLoss((SemWLS(semobserved; meanstructure = true),))
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

solution_ml = sem_fit(model_ml)
@test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01) 

solution_ls = sem_fit(model_ls)
@test SEM.compare_estimates(par_ls.est[par_order], solution_ls.minimizer, 0.01)

############################################################################
### test gradients
############################################################################

@testset "ml_gradients_meanstructure" begin
    grad = similar(start_val_ml)
    grad .= 0.0
    model_ml(start_val_ml, 1.0, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ml(x, 1.0, nothing, nothing), start_val_ml)
    grad .= 0.0
    model_ml(start_val_ml, nothing, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ml(x, 1.0, nothing, nothing), start_val_ml)
end

@testset "ls_gradients_meanstructure" begin
    grad = similar(start_val_ls)
    grad .= 0.0
    model_ls(start_val_ls, 1.0, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ls(x, 1.0, nothing, nothing), start_val_ls)
    grad .= 0.0
    model_ls(start_val_ls, nothing, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ls(x, 1.0, nothing, nothing), start_val_ls)
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
loss_ml = SemLoss((SEM.SemFIML(semobserved, 1.0, similar(start_val_ml)),))

# imply
imply_ml = RAMSymbolic(A, S, F, x, start_val_ml; M = M)

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
model_ml = SemFiniteDiff(semobserved, imply_ml, loss_ml, diff, false)

############################################################################
### test gradients
############################################################################

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

############################################################################
### test solution
############################################################################

solution_ml = sem_fit(model_ml)
@test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)