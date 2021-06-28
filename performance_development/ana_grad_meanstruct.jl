using sem, Arrow, ModelingToolkit, LinearAlgebra,
    Optim, SparseArrays, Test, LineSearches, DataFrames, Statistics

cd("./test")
## Observed Data
three_path_dat = DataFrame(Arrow.Table("comparisons/three_path_dat.arrow"))
three_path_par = DataFrame(Arrow.Table("comparisons/three_path_mean_par.arrow"))
three_path_start = DataFrame(Arrow.Table("comparisons/three_path_start.arrow"))

semobserved = SemObsCommon(data = Matrix{Float64}(three_path_dat); meanstructure = true)

diff_fin = SemFiniteDiff(BFGS(), Optim.Options(
    ;f_tol = 1e-10, 
    x_tol = 1.5e-8))

## Model definition
@ModelingToolkit.variables x[1:31], mₓ, my[1:8]

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

M = [mₓ, mₓ, mₓ, my[1:5]..., 3.0, my[7:8]..., 0.0, 0.0, 0.0]

#M = sparse(M)

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)

start_val = vcat(
    vec(var(Matrix(three_path_dat), dims = 1))./2,
    fill(0.05, 3),
    fill(0.0, 6),
    fill(1.0, 8),
    fill(0, 3),
    [mean(Matrix(three_path_dat), dims = 1)...][[1, 4:8..., 10:11...]]
    )

loss = Loss([SemML(semobserved, [0.0], similar(start_val))])

imply = ImplySymbolic(A, S, F, [x..., mₓ, my[1:5]..., my[7:8]...], start_val; M = M)

model_fin = Sem(semobserved, imply, loss, diff_fin)

solution_fin = sem_fit(model_fin)

par_order = [collect(25:38); collect(15:20); 2;3; 5;6;7; collect(9:14); 21;
    collect(39:45)]

all(
    abs.(solution_fin.minimizer .- three_path_par.est[par_order]
        ) .< 0.05*abs.(three_path_par.est[par_order]))

#start_lav = three_path_start.start[par_order]

# diff
grad_ml = sem.∇SemML(A, S, F, [x..., mₓ, my[1:5]..., my[7:8]...], start_val; M = M)       

diff_ana = 
    SemAnalyticDiff(
        BFGS(), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8),
            (grad_ml,))  
            
model_ana = Sem(semobserved, imply, loss, diff_ana)

solution_ana = sem_fit(model_ana)

all(
    abs.(solution_ana.minimizer .- three_path_par.est[par_order]
        ) .< 0.05*abs.(three_path_par.est[par_order]))