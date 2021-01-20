using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff,
    BenchmarkTools, ProfileView

## Observed Data #############################################################

data_small = Feather.read("test/comparisons/data_unique_small.feather")
data_big = Feather.read("test/comparisons/data_unique_big.feather")
data_huge = Feather.read("test/comparisons/data_unique_huge.feather")

data_def_small = Feather.read("test/comparisons/data_def_unique_small.feather")
data_def_big = Feather.read("test/comparisons/data_def_unique_big.feather")
data_def_huge = Feather.read("test/comparisons/data_def_unique_huge.feather")

pars_small = Feather.read("test/comparisons/pars_unique_small.feather")
pars_big = Feather.read("test/comparisons/pars_unique_big.feather")
pars_huge = Feather.read("test/comparisons/pars_unique_huge.feather")


## small ############################################################

semobserved_small = 
    SemObsCommon(data = Matrix(data_small); meanstructure = true)

diff_fin_small = SemFiniteDiff(BFGS(), Optim.Options())

@ModelingToolkit.variables x[1:7], m[1:2], load_t[1:4]

S =[x[1]  0     0     0     0     0     
    0     x[2]  0     0     0     0     
    0     0     x[3]  0     0     0     
    0     0     0     x[4]  0     0     
    0     0     0     0     x[5]  x[7]    
    0     0     0     0     x[7]  x[6]]

F =[1.0 0 0 0 0 0
    0 1 0 0 0 0
    0 0 1 0 0 0
    0 0 0 1 0 0]

A =[0  0  0  0  1.0  load_t[1]
    0  0  0  0  1  load_t[2]
    0  0  0  0  1  load_t[3] 
    0  0  0  0  1  load_t[4]
    0  0  0  0  0  0 
    0  0  0  0  0  0]

M = [0.0, 0, 0, 0, m[1:2]...]

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)

data_def_small = Matrix(data_def_small)  

start_val = [1.0, 1, 1, 1, 1, 1, 1, 1.0, 1.0]

imply_small = ImplySymbolicDefinition(
    A, 
    S,
    F, 
    M, 
    [x[1:7]..., m[1:2]...], 
    load_t,
    start_val,
    data_def_small
)

loss_small = Loss([SemDefinition(semobserved_small, imply_small, 0.0, 0.0)])

model_fin_small = Sem(semobserved_small, imply_small, loss_small, diff_fin_small)

solution_small = sem_fit(model_fin_small)

par_order = [collect(1:5); 7; 6; 8; 9]

all(
    abs.(solution_small.minimizer .- pars_small.Estimate[par_order]
        ) .< 0.001*abs.(pars_small.Estimate[par_order]))

@benchmark sem_fit(model_fin_small)

## big ############################################################

semobserved_big = 
    SemObsCommon(data = Matrix(data_big); meanstructure = true)

diff_fin_big = SemFiniteDiff(LBFGS(), Optim.Options(x_tol = 1e-9))

@ModelingToolkit.variables x[1:18], m[1:2], load_t[1:15]

F = zeros(15, 17)

diag(F)
F[diagind(F)] .= 1.0

M = zeros(Expression, 17, 1)
M .= [zeros(15)..., m[1:2]...]
M = sparse(M)

#S
Ind = [1:17..., 16, 17]; J = [1:17..., 17, 16]; V = [x[1:17]..., x[18], x[18]]
S = sparse(Ind, J, V)

#F
F = sparse(F)

#A
Ind = [1:15..., 1:15...]; J = [fill(16, 15)..., fill(17, 15)...]; V = [ones(15)..., load_t...]
A = sparse(Ind, J, V, 17, 17)

data_def_big = Matrix(data_def_big)  

start_val = ones(20) #[ones(15)..., 0.05, 0.05, 0.0, 1.0, 1.0]

imply_big = ImplySymbolicDefinition(
    A, 
    S,
    F, 
    M, 
    [x[1:18]..., m[1:2]...], 
    load_t,
    start_val,
    data_def_big
)

loss_big = Loss([SemDefinition(semobserved_big, imply_big, 0.0, 0.0)])

model_fin_big = Sem(semobserved_big, imply_big, loss_big, diff_fin_big)

solution_big = sem_fit(model_fin_big)

par_order = [collect(1:16); 18; 17; 19; 20]

all(
    abs.(solution_big.minimizer .- pars_big.Estimate[par_order]
        ) .< 0.001*abs.(pars_big.Estimate[par_order]))

@benchmark sem_fit(model_fin_big)

## huge ############################################################

semobserved_huge = 
    SemObsCommon(data = Matrix(data_huge); meanstructure = true)

diff_fin_huge = SemFiniteDiff(BFGS(), Optim.Options())

@ModelingToolkit.variables x[1:33], m[1:2], load_t[1:30]

F = zeros(30, 32)

diag(F)
F[diagind(F)] .= 1.0

M = [zeros(30)..., m[1:2]...]

#S
Ind = [1:32..., 31, 32]; J = [1:32..., 32, 31]; 
V = [x[1:32]..., x[33], x[33]];

S = sparse(Ind, J, V)

#F
F = sparse(F)

#A
Ind = [1:30..., 1:30...]; J = [fill(31, 30)..., fill(32, 30)...]; 
V = [ones(30)..., load_t...];
A = sparse(Ind, J, V, 32, 32)

data_def_huge = Matrix(data_def_huge)  

start_val = ones(35)#[ones(30)..., 0.05, 0.05, 0.0, 1.0, 1.0]

imply_huge = ImplySymbolicDefinition(
    A, 
    S,
    F, 
    M, 
    [x[1:33]..., m[1:2]...], 
    load_t,
    start_val,
    data_def_huge
)

loss_huge = Loss([SemDefinition(semobserved_huge, imply_huge, 0.0, 0.0)])

model_fin_huge = Sem(semobserved_huge, imply_huge, loss_huge, diff_fin_huge)

solution_huge = sem_fit(model_fin_huge)

par_order = [collect(1:31); 33; 32; 34; 35]

all(
    abs.(solution_huge.minimizer .- pars_huge.Estimate[par_order]
        ) .< 0.001*abs.(pars_huge.Estimate[par_order]))

@benchmark sem_fit(model_fin_huge)

## benchmark #######################################################
start_val = [ones(15)..., 0.05, 0.05, 0.0, 1.0, 1.0]


start_val = [ones(30)..., 0.05, 0.05, 0.0, 1.0, 1.0]


@code_warntype model_fin_big.imply(start_val)

@code_warntype model_fin_big(start_val)

@code_warntype model_fin_big.loss.functions[1](start_val, model_fin_big)


@benchmark model_fin_big.imply($start_val)

@benchmark model_fin_big(start_val)
ProfileView.@profview model_fin_big(start_val)

@benchmark model_fin_big.loss.functions[1](start_val, model_fin_big)

@benchmark model_fin_huge.imply($start_val)

@benchmark model_fin_huge($start_val)

@benchmark model_fin_big.loss.functions[1]($start_val, $model_fin_big)
#1.5ms

function profile_test(n)
    for i = 1:n
        A = randn(100,100,20)
        m = maximum(A)
        Am = mapslices(sum, A; dims=2)
        B = A[:,:,5]
        Bsort = mapslices(sort, B; dims=1)
        b = rand(100)
        C = B.*b
    end
end
# run once to trigger compilation (ignore this one)

@btime profile_test(10)
@btime to_prof(100, start_val, model_fin_big)

function to_prof(n, par, model)
    for i = 1:n
        model.loss.functions[1](par, model)
    end
end

ProfileView.@profview to_prof(1000, start_val, model_fin_big)

using LinearAlgebra, BenchmarkTools
BLAS.vendor()

using CUDA, MKL

a = rand(30,30); a = a'*a

BLAS.set_num_threads(8)

set_max_threads(n) = ccall((:mkl_set_num_threads, libmkl_rt), Cvoid, (Ptr{Int32},), Ref(Int32(n)));
set_max_threads(3)
MKL.typemin()

@benchmark inv(a)


a = CUDA.rand(Float64, 30, 30)
a32 = CUDA.rand(30, 30)

a32 isa StridedCuArray

inv(a32)

b = rand(30,30)
b32 = rand(Float32, 30 , 30)

@benchmark inv(b)

@benchmark inv(b32)

a = rand(5)


### bounds

a = rand(30,30); a = a*a'
matvec = [a for i = 1:500]
safe = copy(matvec)
pre = similar(matvec)

copyto!(safe, matvec)

cholvec = cholesky.(matvec)

function myf(pre, matvec)
    @inbounds @fastmath for i = 1:size(pre, 1)
        pre[i] = matvec[i]^2
    end
end

function myf2(pre, matvec)
    @inbounds @fastmath for i = 1:size(pre, 1)
        pre[i] = matvec[i]^2
    end
end

@benchmark myf(pre, cholvec)

@benchmark myf2(pre, matvec)


# 0.2720108 secs vs 100ms
# 36.52637 secs vs 30s (21s)
# 156.6648 secs vs 157s