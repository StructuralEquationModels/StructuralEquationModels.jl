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


### diff objects ##########################################################

## Tolerances
# what about x_tol and g_tol?
omx_abstol = 6.3e-12
objective_value = 41856.14242753569
objective_value = 2.272870e+04
rel_tol = omx_abstol/(objective_value+1)

f_tol = rel_tol #6.3e-12
time_limit = 40

## initial step length
#InitialPrevious (Use the step length from the previous optimization iteration)
#InitialStatic (Use the same initial step length each time) (scaled = TRUE?)
#InitialHagerZhang (Taken from Hager and Zhang, 2006)
#InitialQuadratic (Propose initial step length based on a quadratic interpolation) (proposed for grad. descent/conj. grad)
#InitialConstantChange (Propose initial step length assuming constant change in step length)

## Line Search
#HagerZhang (Taken from the Conjugate Gradient implementation by Hager and Zhang, 2006)
#MoreThuente (From the algorithm in More and Thuente, 1994)
#BackTracking (Described in Nocedal and Wright, 2006) (quadratic or cubic)
#StrongWolfe (Nocedal and Wright)
#Static (Takes the proposed initial step length.)

step_length = (InitialPrevious, InitialStatic, InitialQuadratic, InitialHagerZhang, InitialConstantChange)
line_search = (HagerZhang, MoreThuente, BackTracking, StrongWolfe, Static)
algo = (BFGS, LBFGS, Newton)


sl_names = String.(Symbol.(step_length))
ls_names = String.(Symbol.(line_search))
algo_names = String.(Symbol.(algo))
m = [10, 50]

#combinations = Array{String, 1}(undef, 25)

#solutions_BFGS = Dict{String, Any}()
#solutions_LBFGS = Dict{String, Any}()
#solutions_newton = Dict{String, Any}()

using DataFrames, CSV

solutions = 
    DataFrame(
        algo = String[],
        line_search = String[], 
        step_length = String[],
        solution = Any[],
        m = Int64[],
        n = Int64[])


#push!(solutions, "hi" => 1.0)

for i = 1:length(step_length)
    for j = 1:length(line_search)
        for k = 1:length(algo)
            if algo[k] == LBFGS
                for l = 1:length(m)
                    m_ = m[l]
                    diff = SemFiniteDiff(
                        algo[k](;
                            m = m_,
                            alphaguess = step_length[i](), 
                            linesearch = line_search[j]()), 
                        Optim.Options(f_tol = f_tol, time_limit = time_limit))
                    model = Sem(semobserved_big, imply_big, loss_big, diff)
                    solution = sem_fit(model)
                    push!(solutions, (algo_names[k], ls_names[j], sl_names[i], solution, m_, size(solutions)[1]+1))
                end
            else
            m_ = 1    
            diff = SemFiniteDiff(
                algo[k](;
                    alphaguess = step_length[i](), 
                    linesearch = line_search[j]()), 
                Optim.Options(f_tol = f_tol, time_limit = time_limit))
            model = Sem(semobserved_big, imply_big, loss_big, diff)
            solution = sem_fit(model)
            push!(solutions, (algo_names[k], ls_names[j], sl_names[i], solution, m_, size(solutions)[1]+1))
            end
        end
    end
end

#times_BFGS = Dict{String, Any}()

function compare_solutions(mypars, truepars, order)
    all(
        abs.(mypars .- truepars[order]
            ) .< 0.001*abs.(truepars[order]))
end

solutions.minimum = getproperty.(solutions.solution, :minimum)
solutions.minimizer = getproperty.(solutions.solution, :minimizer)
solutions.time = getproperty.(solutions.solution, :time_run)
solutions.f_calls = getproperty.(solutions.solution, :f_calls)
solutions.g_calls = getproperty.(solutions.solution, :g_calls)
solutions.h_calls = getproperty.(solutions.solution, :h_calls) 
solutions.truepars = 
    broadcast(x -> compare_solutions(
        x, 
        pars_big.Estimate, 
        par_order), solutions.minimizer)

select!(solutions, #Not(:solution), 
    Not(:minimizer))
        
Feather.write(
    "test/comparisons/benchmark_small.feather",
    solutions)

CSV.write("test/comparisons/benchmark_big.csv", solutions)



### benchmarks ############################################################


model_fin_small = Sem(semobserved_small, imply_small, loss_small, diff_fin_small)

solution_small = sem_fit(model_fin_small)

par_order = [collect(1:5); 7; 6; 8; 9]

all(
    abs.(solution_small.minimizer .- pars_small.Estimate[par_order]
        ) .< 0.001*abs.(pars_small.Estimate[par_order]))

@benchmark sem_fit(model_fin_small)

##

diff_fin_big = SemFiniteDiff(
    LBFGS(), 
    Optim.Options(f_tol = f_tol))#, time_limit = 20))

diff_fin_big = SemFiniteDiff(
    BFGS(alphaguess = InitialHagerZhang(), 
        linesearch = HagerZhang()), 
    Optim.Options(f_tol = f_tol, time_limit = time_limit))

model_fin_big = Sem(semobserved_big, imply_big, loss_big, diff_fin_big)

solution_big = sem_fit(model_fin_big)

par_order = [collect(1:16); 18; 17; 19; 20]

all(
    abs.(solution_big.minimizer .- pars_big.Estimate[par_order]
        ) .< 0.001*abs.(pars_big.Estimate[par_order]))

@benchmark sem_fit(model_fin_big)

##
diff_fin_huge = SemFiniteDiff(
    LBFGS(; m = 50,
    alphaguess = InitialHagerZhang()), 
    Optim.Options(f_tol = f_tol))

model_fin_huge = Sem(semobserved_huge, imply_huge, loss_huge, diff_fin_huge)

solution_huge = sem_fit(model_fin_huge)

par_order = [collect(1:31); 33; 32; 34; 35]

all(
    abs.(solution_huge.minimizer .- pars_huge.Estimate[par_order]
        ) .< 0.001*abs.(pars_huge.Estimate[par_order]))

@benchmark sem_fit(model_fin_huge)
