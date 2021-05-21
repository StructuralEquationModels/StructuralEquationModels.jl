using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    SparseArrays, BenchmarkTools, Optim

## Observed Data
dat = Feather.read("test/comparisons/reg_1.feather")

semobserved = SemObsCommon(data = Matrix(dat))

f_tol = 2.86e-13
diff_fin = SemFiniteDiff(BFGS(), Optim.Options(;f_tol = f_tol))

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
       xind += 1
    end
end

start_val = vcat(
    fill(1, 2),
    fill(0.5, 3),
    fill(0.1, 19),
    fill(1,2),
    fill(1,19),
    fill(0,19),
    fill(0.5, 6),
    fill(0.05, 2),
    fill(0, 171)
    )

loss = Loss([SemML(semobserved, [0.0], similar(start_val))])

imply = ImplySymbolic(A, S, F, x, start_val)

model_fin = Sem(semobserved, imply, loss, diff_fin)

solution_fin = sem_fit(model_fin)

using NLOpt
diff_nlopt = SemFiniteDiff(:LD_LBFGS, nothing)
model_nlopt = Sem(semobserved, imply, loss, diff_nlopt)
solution_nlopt = sem.sem_fit_nlopt(model_nlopt)

par_order = [collect(21:34); collect(15:20); 2;3; 5;6;7; collect(9:14)]

all(
    abs.(solution_fin.minimizer .- three_path_par.est[par_order]
        ) .< 0.05*abs.(three_path_par.est[par_order]))


##### regularized
ridge = sem.SemRidge(0.1, 21:28)

loss = Loss([SemML(semobserved, [0.0], similar(start_val)), ridge])

model_fin = Sem(semobserved, imply, loss, diff_fin)

solution_fin_reg = sem_fit(model_fin)

all(
    abs.(solution_fin_reg.minimizer .- three_path_par.est[par_order]
        ) .< 0.05*abs.(three_path_par.est[par_order]))

