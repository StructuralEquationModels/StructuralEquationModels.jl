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

# diff
diff = 
    SemDiffOptim(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),# m = 100), 
        #P = 0.5*inv(H0),
        #precondprep = (P, x) -> 0.5*inv(FiniteDiff.finite_difference_hessian(model_ls_ana, x))), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8))

### start values
par_order = [collect(7:21); collect(1:6); collect(28:42)]

####################################################################
# ML estimation
####################################################################

start_val_ml = Vector{Float64}(par_ml.start[par_order])

# loss
loss_ml_g1 = SemLoss((SemML(semobserved_g1, 1.0, similar(start_val_ml)),))
loss_ml_g2 = SemLoss((SemML(semobserved_g2, 1.0, similar(start_val_ml)),))

# imply
imply_ml_g1 = RAMSymbolic(A, S1, F, x, start_val_ml)
imply_ml_g2 = RAMSymbolic(A, S2, F, x, start_val_ml)

# models
model_ml_g1 = Sem(semobserved_g1, imply_ml_g1, loss_ml_g1, SemDiffOptim(nothing, nothing))
model_ml_g2 = Sem(semobserved_g2, imply_ml_g2, loss_ml_g2, SemDiffOptim(nothing, nothing))

model_ml_multigroup = SemEnsemble((model_ml_g1, model_ml_g2), diff, start_val_ml)


############################################################################
### test gradients
############################################################################

using FiniteDiff

@testset "ml_gradients" begin
    grad = similar(start_val_ml)
    grad .= 0.0
    model_ml_multigroup(start_val_ml, 1.0, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ml_multigroup(x, 1.0, nothing, nothing), start_val_ml)
    grad .= 0.0
    model_ml_multigroup(start_val_ml, nothing, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ml_multigroup(x, 1.0, nothing, nothing), start_val_ml)
end

ProfileView.@profview prof_mg(100000, start_val_ml, 1.0, nothing, nothing)

function prof_mg(n, par, F, G, H)
    for i in 1:n
        model_ml_multigroup(par, F, G, H)
    end
end

semtuple = model_ml_multigroup.sems
@benchmark model_ml_multigroup(start_val_ml, 1.0, nothing, nothing)

@code_warntype model_ml_multigroup(start_val_ml, 1.0, nothing, nothing)

function mytupind(semtuple, n, par)
    F = mapreduce(model -> model(par, 1.0, nothing, nothing), +, semtuple)
    #for i in 1:n
    #    sem = semtuple.sems[i]
    #end
    #F+= semtuple[1](par, 1.0, nothing, nothing)
    #F+= semtuple[2](par, 1.0, nothing, nothing)
    return F
end

myf = sin

function wrap(sem, par, F, G, H)
    return sem(par, F, G, H)
end

mytupind(semtuple, 2, start_val_ml)

@benchmark mytupind(semtuple, 2, start_val_ml)
@code_warntype mytupind(semtuple, 2, start_val_ml)

# fit
solution_ml = sem_fit(model_ml_multigroup)
@test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)

@benchmark solution_ml = sem_fit(model_ml_multigroup)

####################################################################
# ML estimation - without Gradients and Hessian
####################################################################

struct UserSemML <: SemLossFunction end

############################################################################
### functors
############################################################################

function (semml::UserSemML)(par, F, G, H, model, weight = nothing)
    a = cholesky(Symmetric(model.imply.Σ); check = false)
    if !isposdef(a)
        if !isnothing(G) stop("analytic gradient of ML is not implemented (yet)") end
        if !isnothing(H) stop("analytic hessian of ML is not implemented (yet)") end
        if !isnothing(F) return Inf end
    end
    ld = logdet(a)
    Σ_inv = LinearAlgebra.inv(a)
    if !isnothing(G) stop("analytic gradient of ML is not implemented (yet)") end
    if !isnothing(H) stop("analytic hessian of ML is not implemented (yet)") end
    if !isnothing(F)
        prod = Σ_inv*model.observed.obs_cov
        F = ld + tr(prod)
        if !isnothing(model.imply.μ) end
        if !isnothing(weight)
            F = weight*F
        end
        return F
    end
end

start_val_ml = Vector{Float64}(par_ml.start[par_order])

# loss
loss_ml_g1 = SemLoss((SemML(semobserved_g1, 1.0, similar(start_val_ml)),))
loss_ml_g2 = SemLoss((UserSemML(),))

# imply
imply_ml_g1 = RAMSymbolic(A, S1, F, x, start_val_ml)
imply_ml_g2 = RAMSymbolic(A, S2, F, x, start_val_ml)

# models
model_ml_g1 = Sem(semobserved_g1, imply_ml_g1, loss_ml_g1, SemDiffOptim(nothing, nothing))
model_ml_g2 = SemFiniteDiff(semobserved_g2, imply_ml_g2, loss_ml_g2, SemDiffOptim(nothing, nothing), false)

model_ml_multigroup = SemEnsemble((model_ml_g1, model_ml_g2), diff, start_val_ml)

model_ml_multigroup(start_val_ml, "give me your F value", nothing, nothing)

grad = similar(start_val_ml)
grad .= 0

model_ml_multigroup(start_val_ml, "give me your F value", grad, nothing)

@testset "ml_gradients" begin
    grad = similar(start_val_ml)
    grad .= 0.0
    model_ml_multigroup(start_val_ml, 1.0, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ml_multigroup(x, 1.0, nothing, nothing), start_val_ml)
    grad .= 0.0
    model_ml_multigroup(start_val_ml, nothing, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ml_multigroup(x, 1.0, nothing, nothing), start_val_ml)
end

# fit
solution_ml = sem_fit(model_ml_multigroup)
@test SEM.compare_estimates(par_ml.est[par_order], solution_ml.minimizer, 0.01)

@benchmark solution_ml = sem_fit(model_ml_multigroup)

####################################################################
# GLS estimation
####################################################################

start_val_ls = Vector{Float64}(par_ls.start[par_order])

loss_ls_g1 = SemLoss((SemWLS(semobserved_g1),))
loss_ls_g2 = SemLoss((SemWLS(semobserved_g2),))

imply_ls_g1 = RAMSymbolic(A, S1, F, x, start_val_ls; vech = true)
imply_ls_g2 = RAMSymbolic(A, S2, F, x, start_val_ls; vech = true)

model_ls_g1 = Sem(semobserved_g1, imply_ls_g1, loss_ls_g1, diff)
model_ls_g2 = Sem(semobserved_g2, imply_ls_g2, loss_ls_g2, diff)

@testset "ls_gradients" begin
    grad = similar(start_val_ls)
    grad .= 0.0
    model_ls(start_val_ls, 1.0, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ls(x, 1.0, nothing, nothing), start_val_ls)
    grad .= 0.0
    model_ls(start_val_ls, nothing, grad, nothing)
    @test grad ≈ FiniteDiff.finite_difference_gradient(x -> model_ls(x, 1.0, nothing, nothing), start_val_ls)
end

solution_ls = sem_fit(model_ls)


@test SEM.compare_estimates(par_ls.est[par_order], solution_ls.minimizer, 0.01)