using sem, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Arrow, DataFrames, LineSearches

miss20_dat = DataFrame(Arrow.Table("test/comparisons/dat_miss20_dat.arrow"))
miss20_par = DataFrame(Arrow.Table("test/comparisons/dat_miss20_par.arrow"))
miss20_par_mean = DataFrame(Arrow.Table("test/comparisons/dat_miss20_par_mean.arrow"))
miss20_mat = Matrix(miss20_dat)
miss20_start = DataFrame(Arrow.Table("test/comparisons/dat_miss20_start.arrow"))
miss20_start_mean = DataFrame(Arrow.Table("test/comparisons/dat_miss20_start_mean.arrow"))

three_path_dat = DataFrame(Arrow.Table("test/comparisons/three_path_dat.arrow"))

diff_fin = 
    SemFiniteDiff(
        LBFGS(
            ;alphaguess = LineSearches.InitialHagerZhang(),
            linesearch = LineSearches.HagerZhang()
        ),
        Optim.Options(;f_tol = 1e-10, x_tol = 1.5e-8)
    )

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

# start_val = vcat(
#     vec(var(Matrix(three_path_dat), dims = 1))./2,
#     fill(0.05, 3),
#     fill(0.0, 6),
#     fill(1.0, 8),
#     fill(0, 3)
#     )

par_order = [collect(21:34); collect(15:20); 2;3; 5;6;7; collect(9:14);
    collect(35:45)]

par_order_mean = [collect(25:38); collect(15:20); 2;3; 5;6;7; collect(9:14); 21;
    collect(39:45)]

#= start_val_mean = vcat(
    vec(var(Matrix(three_path_dat), dims = 1))./2,
    fill(0.05, 3),
    fill(0.0, 6),
    fill(1.0, 8),
    fill(0, 3),
    [mean(Matrix(three_path_dat), dims = 1)...][[1, 4:8..., 10:11...]]
    ) =#
 
start_val_mean = convert(Vector{Float64}, miss20_start_mean.est[par_order_mean])

#= start_val_mean_free = vcat(
    vec(var(Matrix(three_path_dat), dims = 1))./2,
    fill(0.05, 3),
    fill(0.0, 6),
    fill(1.0, 8),
    fill(0, 3),
    [mean(Matrix(three_path_dat), dims = 1)...]
) =#

start_val_mean_free = convert(Vector{Float64}, miss20_start.est[par_order])

semobserved = SemObsMissing(miss20_mat)

imply = ImplySymbolic(A, S, F, [x..., m...], start_val_mean_free; M = M_free)
imply_mean = ImplySymbolic(A, S, F, [x..., mₓ, my[1:5]..., my[7:8]...], start_val_mean; M = M)

loss = Loss([SemFIML(semobserved, imply, [0.0], start_val_mean_free)])
loss_mean = Loss([SemFIML(semobserved, imply_mean, [0.0], start_val_mean)])

model_fin = Sem(semobserved, imply, loss, diff_fin)
model_fin_mean = Sem(semobserved, imply_mean, loss_mean, diff_fin)

solution_fin = sem_fit(model_fin)
solution_fin_mean = sem_fit(model_fin_mean)

pattern = model_fin.observed.patterns[55]
inv(model_fin.imply.imp_cov[pattern, pattern]) ≈ model_fin.loss.functions[1].inverses[55]
model_fin.loss.functions[1].logdets[55] ≈ logdet(model_fin.imply.imp_cov[pattern, pattern])

all(
    abs.(solution_fin.minimizer .- miss20_par.est[par_order]
        ) .< 0.05*abs.(miss20_par.est[par_order]))
all(
    abs.(solution_fin_mean.minimizer .- miss20_par_mean.est[par_order_mean]
        ) .< 0.05*abs.(miss20_par.est[par_order_mean]))


using FiniteDiff

grad_diff = FiniteDiff.finite_difference_gradient(model_fin, start_val_mean_free)

grad_fiml = sem.∇SemFIML(semobserved, imply, A, S, F, [x..., m...], start_val_mean_free; M = M_free)

diff_ana = 
    SemAnalyticDiff(
        LBFGS(
            alphaguess = LineSearches.InitialHagerZhang(),
            linesearch = LineSearches.HagerZhang()
        ), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8),
            (grad_fiml,))  
            
model_ana = Sem(semobserved, imply, loss, diff_ana)

grad = zeros(42)

model_ana(start_val_mean_free, grad)

all(
    abs.(grad .- grad_diff
        ) .< 0.001*abs.(grad_diff))
#solution_ana = sem_fit(model_ana)

solution_ana = sem_fit(model_ana)

all(
    abs.(solution_ana.minimizer .- miss20_par.est[par_order]
        ) .< 0.05*abs.(miss20_par.est[par_order]))

using ProfileView, Profile

ProfileView.@profview sem_fit(model_ana)

Profile.clear(); sem_fit(model_ana); ProfileView.view()

