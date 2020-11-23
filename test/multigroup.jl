using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff

## Observed Data
three_path_dat = Feather.read("test/comparisons/three_path_dat.feather")
three_path_dat_loadeq = Feather.read("test/comparisons/three_path_loadeq_dat.feather")
three_path_par = Feather.read("test/comparisons/three_path_par.feather")
three_path_par_alleq = Feather.read("test/comparisons/three_path_2_par.feather")
three_path_par_loadeq = Feather.read("test/comparisons/three_path_loadeq_par.feather")
three_path_start = Feather.read("test/comparisons/three_path_start.feather")
testdat = Feather.read("test/comparisons/testdat.feather")

## Model definition
@variables x[1:31]

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

start_val = vcat(
    vec(var(Matrix(three_path_dat), dims = 1))./2,
    fill(0.05, 3),
    fill(0.0, 6),
    fill(1.0, 8),
    fill(0, 3)
    )

semobserved = SemObsCommon(data = Matrix(three_path_dat))

diff_fin = SemFiniteDiff(
    BFGS(
        ;alphaguess = LineSearches.InitialStatic(;scaled = true),
        linesearch = LineSearches.Static()),
    Optim.Options()
    )

# diff_fin = SemFiniteDiff(
#     LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled = true),
#         linesearch = LineSearches.Static()),
#     Optim.Options(;g_tol = 0.001)
#     )

loss = Loss([SemML(semobserved, [0.0], similar(start_val))])

imply = ImplySymbolic(A, S, F, x, start_val)

rowind = [1:25, collect(26:50), collect(51:75)]

rowind_loadeq = [1:40, 41:80, 81:150]

differ_group = [
    fill(1, 31),
    vcat(fill(1, 14), fill(2, 17)),
    vcat(fill(1, 7), fill(2, 7), fill(3, 10), fill(2, 7))
    ]

differ_group_2 = [
    fill(1, 31),
    fill(1, 31),
    fill(1, 31)
    ]

differ_group_loadeq = [
    fill(1, 31),
    vcat(fill(2, 20), fill(1, 11)),
    vcat(fill(3, 20), fill(1, 11))
    ]


obs_list = get_observed(rowind, three_path_dat, SemObsCommon)
obs_list_loadeq = get_observed(rowind_loadeq, three_path_dat_loadeq, SemObsCommon)

sem_mg, start_val_mg = MGSem(three_path_dat, start_val, differ_group_2,
    rowind, obs_list, imply, loss, diff_fin)

sem_mg_loadeq, start_val_loadeq = MGSem(three_path_dat_loadeq, start_val,
    differ_group_loadeq, rowind_loadeq, obs_list_loadeq, imply,
    loss, diff_fin)

sem_mg_loadeq

names(three_path_dat_loadeq)

@code_warntype sem_mg(start_val_mg)

solution = sem_fit(sem_mg, start_val_mg)

solution_loadeq = sem_fit(sem_mg_loadeq, start_val_loadeq)

@profile (for i = 1:100000 sem_mg(start_val_mg) end)

model_fin = Sem(semobserved, imply, loss, diff_fin)

sem_fit(model_fin)


##
three_path_par_alleq = three_path_par_alleq[1:34, :]

par_order_alleq = [collect(21:34); collect(15:20); 2; 3; 5; 6; 7; collect(9:14)]

three_path_par_alleq.est[par_order_alleq]

solution.minimizer

all(
    abs.(solution.minimizer .- three_path_par_alleq.est[par_order_alleq]
        ) .< 0.05*abs.(three_path_par_alleq.est[par_order_alleq]))

##
par_old = copy(par_loadeq)

par_loadeq == par_old

par_loadeq =
    solution_loadeq.minimizer[vcat(sem_mg_loadeq.par_subsets...)]

est_loadeq = three_path_par_loadeq.est[vcat(par_order_alleq, 34 .+ par_order_alleq,
    68 .+ par_order_alleq)]

all(
    abs.(par_loadeq .- est_loadeq
        ) .< 0.05*abs.(est_loadeq))

findmax(abs.(par_loadeq .- est_loadeq))

variab = rand(5)

subsets = trues(10, 10)
sub_vec = [subsets[i, :] for i in 1:size(subsets, 1)]
par = rand(10)

card_vec = [findall(sub_vec[i]) for i in 1:10]

function tp_inner(par)
end

# function tp_outer(par, ind)
#     @views for i in 1:size(ind, 2)
#         tp_inner(par[ind[i, :]])
#     end
# end

function tp_outer_vec(par, ind)
    for i in 1:length(ind)
        tp_inner(par[ind[i]])
    end
end

index = Array{Float64, 1}(undef, 10)

function tp_outer_vec_2(par, ind, index)
    for i in 1:length(ind)
        tp_inner(view(par, ind[i]))
    end
end

function tp_outer_vec_3(par, ind, index)
    @views for i in 1:length(ind)
        tp_inner(par[ind[i]])
    end
end

function tp_outer_vec_4(par, ind, index)
    for i in 1:length(ind)
        index = par[ind[i]]
        tp_inner(index)
    end
end

@benchmark tp_outer($par, $subsets)
@benchmark tp_outer_vec($par, $sub_vec)
@benchmark tp_outer_vec_2($par, $card_vec, $index)
@benchmark tp_outer_vec_3($par, $card_vec, $index)
@benchmark tp_outer_vec_4($par, $sub_vec, $index)

view(par, )

bitrand(2,2)

mat = falses(2, 2)

mat[1] = true

mat

sem.parsubset(differ_group, start_val)

using Optim, ForwardDiff, FiniteDiff

f(x) = sum(abs, x)

function Δf(x)
    grad = sign.(x)
    return grad
end


optimize(f, start2, BFGS(); autodiff = :forward)

optimize(f, start2, BFGS())

optimize(f, Δf, start2, BFGS(); inplace = false)

start = [0.0, 0]
start2 = [5.0, -15]

FiniteDiff.finite_difference_gradient(f, start)

sign(0.0)

sign.(start2)

x = rand(5)

(x[i] for i = 1:5)

sol.minimizer - sol2.minimizer


function myf(A)
    d = det(A)
    return d
end

function myf_outer(A)
    F = zero(eltype(A[1]))
    for i in 1:length(A)
        F += myf(A[i])
    end
    return F
end

A = [rand(10,10), rand(10,10), rand(10,10)]

B = rand(10,10)

@benchmark myf(B)

@benchmark myf_outer(A)

@code_warntype myf_outer(A)
