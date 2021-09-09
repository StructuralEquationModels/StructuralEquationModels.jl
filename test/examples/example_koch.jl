using sem, Arrow, ModelingToolkit, LinearAlgebra, SparseArrays, DataFrames, Optim, LineSearches

cd("test")

## Observed Data
dat = DataFrame(Arrow.Table("comparisons/reg_1.arrow"))
par = DataFrame(Arrow.Table("comparisons/reg_1_par.arrow"))
start_lav = DataFrame(Arrow.Table("comparisons/reg_1_start.arrow"))

# observed
semobserved = SemObsCommon(data = Matrix{Float64}(dat))

#diff
diff_fin = SemFiniteDiff(
    LBFGS(),
    Optim.Options(;f_tol = 1e-10, x_tol = 1.5e-8))

## Model definition
@ModelingToolkit.variables x[1:243]

#F
Ind = collect(1:25)
Jnd = collect(1:25)
V = fill(1,25)
F = sparse(Ind, Jnd, V, 25, 46)

#A
Ind = [collect(2:25); 5; collect(6:25); fill(27, 19); 1; 4]
Jnd = [fill(26, 24); 27; collect(27:46); collect(28:46); 26; 27]
V =[x[1:64]; 1.0; 1.0]
A = sparse(Ind, Jnd, V, 46, 46)

#S
Ind = [collect(1:6); 26; 27; collect(28:46)]
Jnd = [collect(1:6); 26; 27; collect(28:46)]
V = [x[65:72]; fill(1.0, 19)]
S = sparse(Ind, Jnd, V, 46, 46)
xind = 73
for i = 28:46
    for j = (i+1):46
       S[i,j] = x[xind]
       S[j,i] = x[xind]
       global xind += 1
    end
end

par_order = [
    collect(2:6); 
    collect(11:29); 
    8;9; 
    collect(49:67);
    collect(106:124); 
    collect(125:130);
    131; 132;
    collect(133:303)]

start_val = Vector{Float64}(start_lav.est[par_order])   

# loss
loss = Loss([SemML(semobserved, [0.0], similar(start_val))])

# imply
imply = ImplySymbolic(A, S, F, x, start_val)
imply_sparse = ImplySparse(A, S, F, x, start_val)
@time imply_common = ImplyCommon(A, S, F, x, start_val)

# model
model_fin = Sem(semobserved, imply, loss, diff_fin)
model_fin_sparse = Sem(semobserved, imply_sparse, loss, diff_fin)
model_fin_common = Sem(semobserved, imply_common, loss, diff_fin)

# fit 
solution_fin = sem_fit(model_fin)
solution_fin_sparse = sem_fit(model_fin_sparse)
solution_fin_common = sem_fit(model_fin_common)

@test all(#
        abs.(solution_fin_sparse.minimizer .- par.est[par_order]
            ) .< 0.05*abs.(par.est[par_order]))

@test all(#
        abs.(solution_fin.minimizer .- par.est[par_order]
            ) .< 0.05*abs.(par.est[par_order]))


grad_ml = sem.∇SemML(A, S, F, x, start_val)       

diff_ana = 
    SemAnalyticDiff(
        LBFGS(
            m = 50,
            alphaguess = InitialHagerZhang(), 
            linesearch = HagerZhang()), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8),
            (grad_ml,))  
            
model_ana = Sem(semobserved, imply, loss, diff_ana)

solution_ana = sem_fit(model_ana)

@test all(
        abs.(solution_ana.minimizer .- par.est[par_order]
            ) .< 0.05*abs.(par.est[par_order]))


a = [1 x[1] 0]

A_dense = Matrix(A)

F_dense = Matrix(F)

S_dense = Matrix(S)

A_type = typeof.(ModelingToolkit.value.(A_dense))

S_type = typeof.(ModelingToolkit.value.(S_dense))

getindex(A)