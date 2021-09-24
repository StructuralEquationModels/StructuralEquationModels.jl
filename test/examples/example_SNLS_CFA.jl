using sem, Arrow, ModelingToolkit, LinearAlgebra, SparseArrays, DataFrames, Optim, LineSearches

cd("test")

## Observed Data
dat = DataFrame(Arrow.Table("comparisons/data_hol.arrow"))
par_ml = DataFrame(Arrow.Table("comparisons/par_hol_ml.arrow"))
par_ls = DataFrame(Arrow.Table("comparisons/par_hol_ls.arrow"))

# observed
semobserved = SemObsCommon(data = Matrix{Float64}(dat))

#diff
diff_fin = SemFiniteDiff(
    BFGS(),
    Optim.Options())

## Model definition
@ModelingToolkit.variables x[1:22]

#x = rand(31)

S = zeros(Num,12,12)
S[diagind(S)] .= x[1:12]
S[7,8] = x[13]; S[8,7] = x[13]
S[10,12] = x[15]; S[12,10] = x[15]
S[11,12] = x[16]; S[12,11] = x[16]
S[10,11] = x[14]; S[11,10] = x[14]

F = zeros(9, 12)
F[diagind(F)] .= 1.0

A =[0  0  0  0  0  0  0  0  0     1     0     0
    0  0  0  0  0  0  0  0  0     x[17] 0     0
    0  0  0  0  0  0  0  0  0     x[18] 0     0
    0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0     0     x[19] 0
    0  0  0  0  0  0  0  0  0     0     x[20] 0
    0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0     0     0     x[21]
    0  0  0  0  0  0  0  0  0     0     0     x[22]
    0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0     0     0     0]

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)
    
par_order = [collect(11:22); collect(10); collect(23:25); 2;3;5;6;8;9]

start_val_ml = Vector{Float64}(par_ml.start[par_order])
start_val_ls = Vector{Float64}(par_ls.start[par_order])
start_val_snlls = Vector{Float64}(par_ls.start[par_order][17:22])

# loss
loss_ml = Loss([SemML(semobserved, [0.0], similar(start_val_ml))])
loss_ls = Loss([sem.SemWLS(semobserved)])
loss_snlls = Loss([sem.SemSWLS(semobserved)])

#start_val_ml = [fill(1.0, 14); fill(0, 6); fill(1, 8); fill(0, 3)]
#start_val_ls = [fill(1.0, 14); fill(0, 6); fill(1, 8); fill(0, 3)]
#start_val_snlls = [fill(1.0, 8); fill(0, 3)]

# imply
imply_ml = ImplySymbolic(A, S, F, x, start_val_ml)
imply_ls = sem.ImplySymbolicWLS(A, S, F, x, start_val_ls)
imply_snlls = sem.ImplySymbolicSWLS(A, S, F, x[17:22], start_val_snlls)

# model
model_ml = Sem(semobserved, imply_ml, loss_ml, diff_fin)
model_ls = Sem(semobserved, imply_ls, loss_ls, diff_fin)
model_snlls = Sem(semobserved, imply_snlls, loss_snlls, diff_fin)

#checks
imply_ml(start_val_ml, model_ml)
imply_ls(start_val_ls, model_ls)

ind = CartesianIndices(imply_ml.imp_cov)
ind = filter(x -> (x[1] >= x[2]), ind)
s = imply_ml.imp_cov[ind]

imply_ls.imp_cov ≈ s

imply_ml(x, model_ml)
imp_cov = F*inv(I-A)*S*transpose(inv(I-A))*transpose(F)
imp_cov ≈ imply_ml.imp_cov

imply_snlls(start_val_snlls, model_snlls)
t = imply_snlls.G*start_val_ls[[1, 2, 3, 4, 15, 5, 16, 17, 6, 18, 7, 19, 8, 9, 20, 10, 11, 12, 13, 14]]
t ≈ imply_ls.imp_cov

V = I
s = cov(Matrix(dat))[tril(trues(size(cov(Matrix(dat)))))]
G = imply_snlls.G

transpose(s)*V*s - transpose(s)*V*G*inv(transpose(G)*V*G)*(transpose(G)*V*s)

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
            abs.(solution_snlls.minimizer .- par_ls.est[par_order][17:22]
                ) .< 0.05*abs.(par_ls.est[par_order][17:22]))


using BenchmarkTools

@benchmark solution_ml = sem_fit(model_ml)
@benchmark solution_ls = sem_fit(model_ls)
@benchmark solution_snlls = sem_fit(model_snlls)

@benchmark model_ml(start_val_ml)
@benchmark model_ls(start_val_ls)
@benchmark model_snlls(start_val_snlls)


a = rand(100,100)
a = a*a'
b = rand(100,1)
b'*a*b

function inv_dot(a, b)
    u = cholesky(a)
    z = u.L\b
    w = u.U\z
    sol = b'*w
    return sol
end

function inv_dot(a, b)
    u = cholesky(a)
    z = u.L\b
    w = u.U\z
    sol = b'*w
    return sol
end

function inv_dot_naive(a, b)
    sol = b'*inv(a)*b
    return sol
end

function inv_dot_dot(a, b)
    sol = dot(b', inv(a), b)
    return sol
end

function inv_solve(a, b)
    right = a\b
    sol = b'*right
    return sol
end

function inv_chol(a, b)
    c = cholesky(a)
    right = c\b
    sol = b'*right
    return sol
end

inv_dot(a, b) ≈ inv_dot_naive(a, b)
inv_dot_dot(a, b) ≈ inv_dot_naive(a, b)[1]
inv_solve(a, b) ≈ inv_dot_naive(a, b)
inv_chol(a, b) ≈ inv_dot_naive(a, b)

@benchmark inv_dot($a, $b)
@benchmark inv_dot_naive($a, $b)
@benchmark inv_dot_dot($a, $b)
@benchmark inv_solve($a, $b)
@benchmark inv_chol($a, $b)

