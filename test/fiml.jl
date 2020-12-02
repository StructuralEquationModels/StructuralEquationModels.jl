using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff

miss20_dat = Feather.read("test/comparisons/dat_miss20_dat.feather")
miss30_dat = Feather.read("test/comparisons/dat_miss30_dat.feather")
miss50_dat = Feather.read("test/comparisons/dat_miss50_dat.feather")

miss20_par = Feather.read("test/comparisons/dat_miss20_par.feather")
miss30_par = Feather.read("test/comparisons/dat_miss30_par.feather")
miss50_par = Feather.read("test/comparisons/dat_miss50_par.feather")

three_path_dat = Feather.read("test/comparisons/three_path_dat.feather")

miss20_mat = Matrix(miss20_dat)


diff_fin = SemFiniteDiff(BFGS(), Optim.Options())

## Model definition
@ModelingToolkit.variables x[1:31], mₓ, my[1:8], m[1:11]

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

M_free = [m[1:11]..., 0.0, 0.0, 0.0]

#S

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
    fill(0, 3)
    )

start_val_mean = vcat(
    vec(var(Matrix(three_path_dat), dims = 1))./2,
    fill(0.05, 3),
    fill(0.0, 6),
    fill(1.0, 8),
    fill(0, 3),
    [mean(Matrix(three_path_dat), dims = 1)...][[1, 4:8..., 10:11...]]
    )

semobserved = SemObsMissing(miss20_mat)
semobserved_mean = SemObsMissing(miss20_mat; meanstructure = true)

loss = Loss([SemFIML(semobserved, [0.0], start_val)])
loss_mean = Loss([SemFIML(semobserved_mean, [0.0], start_val_mean)])

imply = ImplySymbolic(A, S, F, x, start_val)
imply_mean = ImplySymbolic(A, S, F, [x..., mₓ, my[1:5]..., my[7:8]...], start_val_mean; M = M)
imply_mean_free = ImplySymbolic(A, S, F, [x..., m...], start_val_mean; M = M_free)

model_fin = Sem(semobserved, imply, loss, diff_fin)
model_fin_mean = Sem(semobserved_mean, imply_mean, loss_mean, diff_fin)
model_fin_mean_free = Sem(semobserved_mean, imply_mean_free, loss_mean, diff_fin)

using BenchmarkTools

@benchmark model_fin(start_val)

model_fin_mean(start_val_mean)

solution_fin = sem_fit(model_fin)
solution_fin = sem_fit(model_fin_mean)
solution_fin = sem_fit(model_fin_mean_free)

par_order = [collect(25:38); collect(15:20); 2;3; 5;6;7; collect(9:14); 21;
    collect(39:45)]

all(
    abs.(solution_fin.minimizer .- three_path_par.est[par_order]
        ) .< 0.05*abs.(solution_fin.minimizer))

#start_lav = three_path_start.start[par_order]

getindex(5.0, [1])
