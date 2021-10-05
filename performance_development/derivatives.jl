using sem, Arrow, ModelingToolkit, LinearAlgebra, 
    SparseArrays, DataFrames, Optim, LineSearches,
    Statistics

cd("test")

## Observed Data
dat = DataFrame(Arrow.Table("comparisons/data_dem.arrow"))
par_ml = DataFrame(Arrow.Table("comparisons/par_dem_ml.arrow"))
par_ls = DataFrame(Arrow.Table("comparisons/par_dem_ls.arrow"))

dat = 
    select(
        dat, 
        [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8])
# observed
semobserved = SemObsCommon(data = Matrix{Float64}(dat))

#diff
diff_fin = SemFiniteDiff(
    BFGS(),
    Optim.Options())

diff_fin_new = SemFiniteDiff(
    BFGS(),
    Optim.Options())

## Model definition
@ModelingToolkit.variables x[1:31]

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
start_val_ls = Vector{Float64}(par_ls.start[par_order])
start_val_snlls = Vector{Float64}(par_ls.start[par_order][21:31])

# loss
loss_ml = Loss([SemML(semobserved, [0.0], similar(start_val_ml))])
loss_ls = Loss([sem.SemWLS(semobserved, [0.0], similar(start_val_ml))])
loss_snlls = Loss([sem.SemSWLS(semobserved, [0.0], similar(start_val_ml))])

# imply
imply_ml = ImplySymbolic(A, S, F, x, start_val_ml)
imply_ls = sem.ImplySymbolicWLS(A, S, F, x, start_val_ls)
imply_snlls = sem.ImplySymbolicSWLS(A, S, F, x[21:31], start_val_snlls)

# model
model_ml = Sem(semobserved, imply_ml, loss_ml, diff_fin)
model_ls = Sem(semobserved, imply_ls, loss_ls, diff_fin)
model_snlls = Sem(semobserved, imply_snlls, loss_snlls, diff_fin_new)

grad_ls = sem.∇SemWLS(loss_ls.functions[1], size(F, 1))


diff_ana = 
    SemAnalyticDiff(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8),
            (grad_ls,))

model_ls_ana = Sem(semobserved, imply_ls, loss_ls, diff_ana)

grad_ml = sem.∇SemML_2()       

diff_ana_ml = 
    SemAnalyticDiff(
        BFGS(;linesearch = BackTracking(order=3), alphaguess = InitialHagerZhang()),
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8),
            (grad_ml,))

model_ml_ana = Sem(semobserved, imply_ml, loss_ml, diff_ana_ml)

# fit
solution_ml = sem_fit(model_ml)
solution_ls = sem_fit(model_ls)
solution_snlls = sem_fit(model_snlls)
solution_ana = sem_fit(model_ls_ana)
solution_ana_ml = sem_fit(model_ml_ana)

all(#
        abs.(solution_ml.minimizer .- par_ml.est[par_order]
            ) .< 0.05*abs.(par_ml.est[par_order]))

all(#
            abs.(solution_ls.minimizer .- par_ls.est[par_order]
                ) .< 0.05*abs.(par_ls.est[par_order]))

all(#
            abs.(solution_snlls.minimizer .- par_ls.est[par_order][21:31]
                ) .< 0.05*abs.(par_ls.est[par_order][21:31]))

all(#
            abs.(solution_ana.minimizer .- par_ls.est[par_order]
                ) .< 0.05*abs.(par_ls.est[par_order]))

all(#
            abs.(solution_ana_ml.minimizer .- par_ml.est[par_order]
                ) .< 0.05*abs.(par_ml.est[par_order]))


using BenchmarkTools, FiniteDiff

# ls gradient
grad = FiniteDiff.finite_difference_gradient(model_ls, start_val_ls)
grad2 = similar(grad)
model_ls_ana.diff(start_val_ls, grad2, model_ls_ana)

grad ≈ grad2

@benchmark model_ls_ana.diff(start_val_ls, grad2, model_ls_ana)
@benchmark FiniteDiff.finite_difference_gradient(model_ls, start_val_ls)

@benchmark sem_fit(model_ls)
@benchmark sem_fit(model_ls_ana)

# ml gradient
grad = FiniteDiff.finite_difference_gradient(model_ml, start_val_ml)
grad2 = similar(grad)
model_ml_ana.diff(start_val_ml, grad2, model_ml_ana)

grad ≈ grad2

@benchmark model_ml_ana.diff(start_val_ml, grad2, model_ml_ana)
@benchmark FiniteDiff.finite_difference_gradient(model_ml, start_val_ml)

@benchmark sem_fit(model_ml)
@benchmark sem_fit(model_ml_ana)

############################## Appendix
# derivatives
function get_Knn(n)
    Knn = zeros(n^2, n^2)
    for i = 1:n
        for j = 1:n
          Knn[i + n*(j - 1), j + n*(i - 1)] = 1
        end
    end
    return Knn
end

function get_Knn_2(n)
    Knn = zeros(n^2, n^2)
    for i = 1:n
        for j = 1:n
            e1 = zeros(n)
            e2 = zeros(n)
            e1[i] = 1; e2[j] = 1
            Knn += kron(e1*e2', e2*e1')
        end
    end
    return Knn
end

get_Knn_2(11) == get_Knn(11)

function Matrix_Jacobian(M, x)
    Mv = vec(M)
    JM = zeros(size(Mv, 1), size(x, 1))
    for i = 1:size(x, 1)
        JM[:, i] .= isequal.(Mv, x[i])
    end
    return JM
end

JS = Matrix_Jacobian(S, x)
JA = Matrix_Jacobian(A, x)
JAT = Matrix_Jacobian(permutedims(A), x)

function elimination_matrix(nobs)
    nobs = Int(nobs)
    n1 = Int(nobs*(nobs+1)*0.5)
    n2 = Int(nobs^2)
    L = zeros(n1, n2)

    for j in 1:nobs
        for i in j:nobs
            u = zeros(n1)
            u[Int((j-1)*nobs + i-0.5*j*(j-1))] = 1
            T = zeros(nobs, nobs)
            T[i, j] = 1
            L += u*transpose(vec(T)) 
        end
    end
    return L
end

function der(x, JS, JA, JAT, V, s, σ)
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

    JS = Matrix_Jacobian(S, x)
    JA = Matrix_Jacobian(A, x)
    JAT = Matrix_Jacobian(permutedims(A), x)

    n = size(A, 1)
    Knn = get_Knn(n)
    A_inv = inv(I-A)
    K = kron(F, F)*kron(A_inv, A_inv)

    In = zeros(n,n); In[diagind(In)] .= 1.0
    J = K*(JS + kron((A_inv*S)', In)*JA + kron(In, S*A_inv')*JAT)

    L = elimination_matrix(size(F, 1))
    
    J = -2*(s-σ)'*V*L*J
    return J
end

par = rand(31)

imply_ls(par, model_ls)
σ = imply_ls.imp_cov
s = loss_ls.functions[1].s
V = loss_ls.functions[1].V

J = der(par, JS, JA, JAT, V, s, σ)

using FiniteDiff

maximum(abs.(FiniteDiff.finite_difference_gradient(model_ls, par) .- J'))

function symbolic_derivative(x, A, S, F)

    JS = Matrix_Jacobian(S, x)
    JA = Matrix_Jacobian(A, x)
    JAT = Matrix_Jacobian(permutedims(A), x)

    JS = sparse(JS)
    JA = sparse(JA)
    JAT = sparse(JAT)

    JS = convert(SparseMatrixCSC{Num, Int64}, JS)
    JA = convert(SparseMatrixCSC{Num, Int64}, JA)
    JAT = convert(SparseMatrixCSC{Num, Int64}, JAT)

    A_inv = sem.neumann_series(A)
    FA = F*A_inv
    K = kron(FA, FA)

    In = oneunit(A)

    J = K*(JS + kron(permutedims(A_inv*S), In)*JA + kron(In, S*permutedims(A_inv))*JAT)
    J = Matrix(J)
    J = simplify.(J)
    J = sparse(J)
    #L = elimination_matrix(size(F, 1))
    #J = -2*(s-σ)'*V*L*J
    return J
end

J = symbolic_derivative(x, A, S, F)

invia = sem.neumann_series(A)

imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)

imp_cov_sym = Array(imp_cov_sym)
imp_cov_sym = ModelingToolkit.simplify.(imp_cov_sym)

J2 = ModelingToolkit.jacobian(vec(imp_cov_sym), x)
J2 = simplify.(J2)
J2 = sparse(J2)

J2 = ModelingToolkit.sparsejacobian(vec(imp_cov_sym), x) 

function identity_kron(A)
    n = size(A, 1)
    n2 = n^2
    res = zeros(size(n2, n2))
    for i in 1:n
        for
end

# fit
solution_ml = sem_fit(model_ml)
solution_ls = sem_fit(model_ls)
solution_snlls = sem_fit(model_snlls)

all(#
        abs.(solution_ml.minimizer .- par_ml.est[par_order]
            ) .< 0.05*abs.(par_ml.est[par_order]))

all(#
            abs.(solution_ls.minimizer .- par_ls.est[par_order]
                ) .< 0.05*abs.(par_ls.est[par_order]))

all(#
            abs.(solution_snlls.minimizer .- par_ls.est[par_order][21:31]
                ) .< 0.05*abs.(par_ls.est[par_order][21:31]))


model_snlls(solution_ls.minimizer[21:31])

t = solution_ls.minimizer[21] .+ collect(-0.5:0.05:0.5)

t_vec = [[t_i; solution_ls.minimizer[22:31]...] for t_i in t]

y = model_snlls.(t_vec)

using Plots

plot(t, y, xticks = t[1:5:21])

function get_fvec(model, pos, solution, iterator)
    t = solution[pos] .+ collect(iterator)
    t_vec = [copy(solution) for i in t]
    for i = 1:size(t, 1)
        t_vec[i][pos] = t[i]
    end
    y = [model(t_vec_i) for t_vec_i in t_vec]
    return plot(t, y, xticks = t[1:5:21])
end

get_fvec(model_snlls, 10, solution_snlls.minimizer, -0.5:0.05:0.5)
    
plots = [get_fvec(model_snlls, i, solution_snlls.minimizer, -0.5:0.05:0.5) for i in 1:11]


plots[11]

plot(plots...)

solution_snlls.minimizer - par_ls.est[par_order][21:31]


using BenchmarkTools

@benchmark solution_ml = sem_fit(model_ml)
@benchmark solution_ls = sem_fit(model_ls)
@benchmark solution_snlls = sem_fit(model_snlls)

@benchmark model_ml(start_val_ml)
@benchmark model_ls(start_val_ls)
@benchmark model_snlls(start_val_snlls)

############### NLopt

diff_nlopt = SemFiniteDiff(
    :LD_LBFGS, 
    nothing)

start_val_snlls = [fill(1.0, 8); fill(0, 3)]
imply_snlls = sem.ImplySymbolicSWLS(A, S, F, x[21:31], start_val_snlls)

model_ml = Sem(semobserved, imply_ml, loss_ml, diff_nlopt)
model_ls = Sem(semobserved, imply_ls, loss_ls, diff_nlopt)
model_snlls = Sem(semobserved, imply_snlls, loss_snlls, diff_nlopt)

solution_ml = 
    sem.sem_fit_nlopt(
        model_ml; 
        lower = fill(-11.0, 31),
        upper = fill(10.0, 31),
        local_algo = :LD_LBFGS)[2]
solution_ls = 
    sem.sem_fit_nlopt(
        model_ls; 
        lower = fill(-11.0, 31),
        upper = fill(10.0, 31))[2]
solution_snlls = 
    sem.sem_fit_nlopt(
        model_snlls;
        lower = fill(-11.0, 11),
        upper = fill(10.0, 11))[2]

all(#
    abs.(solution_ml .- par_ml.est[par_order]
        ) .< 0.05*abs.(par_ml.est[par_order]))

all(#
    abs.(solution_ls .- par_ls.est[par_order]
        ) .< 0.05*abs.(par_ls.est[par_order]))

all(#
    abs.(solution_snlls .- par_ls.est[par_order][21:31]
        ) .< 0.05*abs.(par_ls.est[par_order][21:31]))



@benchmark solution_ls = 
    sem.sem_fit_nlopt(
        model_ls; 
        lower = fill(-11.0, 31),
        upper = fill(10.0, 31))[2]

