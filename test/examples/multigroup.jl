using SEM, CSV, DataFrames, SparseArrays, Symbolics, LineSearches, Optim, Test, LinearAlgebra

############################################################################
### observed data
############################################################################

dat = DataFrame(CSV.File("examples/data/data_multigroup.csv"))
par_ml = DataFrame(CSV.File("examples/data/par_multigroup_ml.csv"))
par_ls = DataFrame(CSV.File("examples/data/par_multigroup_ls.csv"))

par_ml = filter(row -> (row.free != 0)&(row.op != "~1"), par_ml)
par_ls = filter(row -> (row.free != 0)&(row.op != "~1"), par_ls)

dat = 
    select(
        dat,
        [:school, :x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9])

dat_g1 = select(filter(row -> row.school == "Pasteur", dat), Not(:school))
dat_g2 = select(filter(row -> row.school == "Grant-White", dat), Not(:school))

dat = select(dat, Not(:school))

# observed
semobserved_g1 = SemObsCommon(data = Matrix{Float64}(dat_g1))
semobserved_g2 = SemObsCommon(data = Matrix{Float64}(dat_g2))

############################################################################
### define models
############################################################################

@Symbolics.variables x[1:36]
x = [x...]

F = zeros(9, 12)
F[diagind(F)] .= 1.0

A = zeros(Num, 12, 12)
A[1, 10] = 1.0; A[4, 11] = 1.0; A[7, 12] = 1.0
A[2:3, 10] .= x[16:17]; A[5:6, 11] .= x[18:19]; A[8:9, 12] .= x[20:21]; 

# group 1
S1 = zeros(Num, 12, 12)
S1[diagind(S1)] .= x[1:12]
S1[10, 11] = x[13]; S1[11, 10] = x[13]
S1[10, 12] = x[14]; S1[12, 10] = x[14]
S1[12, 11] = x[15]; S1[11, 12] = x[15]

# group 2
S2 = zeros(Num, 12, 12)
S2[diagind(S2)] .= x[22:33]
S2[10, 11] = x[34]; S2[11, 10] = x[34]
S2[10, 12] = x[35]; S2[12, 10] = x[35]
S2[12, 11] = x[36]; S2[11, 12] = x[36]

# S
S1 = sparse(S1)
S2 = sparse(S2)

#F
F = sparse(F)

#A
A = sparse(A)


### start values
par_order = [collect(7:21); collect(1:6); collect(28:42)]
start_val_ml = Vector{Float64}(par_ml.start[par_order])
start_val_ls = Vector{Float64}(par_ls.start[par_order])
# start_val_snlls = Vector{Float64}(par_ls.start[par_order][21:31])

####################################################################
# ML estimation
####################################################################

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

# fit
solution_ml = sem_fit(model_ml)
@test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)

solution_ls = sem_fit(model_ls)
@test SEM.compare_estimates(par_ls.est[par_order], solution_ls.minimizer, 0.01)