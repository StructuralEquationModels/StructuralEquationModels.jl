using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff


## Observed Data #############################################################

growth_dat = Feather.read("test/comparisons/growth_dat.feather")
growth_dat_miss30 = Feather.read("test/comparisons/growth_dat_miss30.feather")
#growth_par = Feather.read("test/comparisons/growth_par.feather")

definition_dat = Feather.read("test/comparisons/definition_dat.feather")
definition_dat_unique = 
    Feather.read("test/comparisons/definition_dat_unique.feather")
definition_par = Feather.read("test/comparisons/definition_par.feather")
definition_par_missing = 
    Feather.read("test/comparisons/definition_par_missing.feather")
definition_par_missing_unique = 
    Feather.read("test/comparisons/definition_par_missing_unique.feather")


##############################################################################

semobserved = SemObsCommon(data = Matrix(growth_dat); meanstructure = true, rowwise = true)

diff_fin = SemFiniteDiff(BFGS(), Optim.Options())

## Model definition ##########################################################

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

data_def = Matrix(definition_dat)  

start_val = [1.0, 1, 1, 1, 0.05, 0.05, 0.0, 1.0, 1.0]

#############################################################################

imply = ImplySymbolicDefinition(
    A, 
    S,
    F, 
    M, 
    [x[1:7]..., m[1:2]...], 
    load_t,
    start_val,
    data_def
)

loss = Loss([SemML(semobserved, imply, 0.0, 0.0)])

model_fin = Sem(semobserved, imply, loss, diff_fin)

solution = sem_fit(model_fin)

par_order = [collect(1:5); 7; 6; 8; 9]

all(
    abs.(solution.minimizer .- definition_par.Estimate[par_order]
        ) .< 0.05*abs.(definition_par.Estimate[par_order]))



## missings ############################################################

data_miss = Matrix(growth_dat_miss30)

semobserved = SemObsMissing(data_miss)   


data_def = data_def[sem.remove_all_missing(data_miss)[2], :]

imply = ImplySymbolicDefinition(
    A, 
    S,
    F, 
    M, 
    [x[1:7]..., m[1:2]...], 
    load_t,
    start_val,
    data_def
    )

loss = Loss([SemFIML(semobserved, imply, 0.0, 0.0)])

model_fin = Sem(semobserved, imply, loss, diff_fin)

solution = sem_fit(model_fin)

par_order = [collect(1:5); 7; 6; 8; 9]

all(
    abs.(solution.minimizer .- definition_par_missing.Estimate[par_order]
        ) .< 0.05*abs.(definition_par_missing.Estimate[par_order]))



## missings with unique defvar patterns #####################################

data_def_unique = Matrix(definition_dat_unique)
data_def_unique = data_def_unique[sem.remove_all_missing(data_miss)[2], :]

imply = ImplySymbolicDefinition(
    A, 
    S,
    F, 
    M, 
    [x[1:7]..., m[1:2]...], 
    load_t,
    start_val,
    data_def_unique
    )

loss = Loss([SemFIML(semobserved, imply, 0.0, 0.0)])

model_fin = Sem(semobserved, imply, loss, diff_fin)

solution = sem_fit(model_fin)

par_order = [collect(1:5); 7; 6; 8; 9]

all(
    abs.(solution.minimizer .- definition_par_missing_unique.Estimate[par_order]
    ) .< 0.05*abs.(definition_par_missing_unique.Estimate[par_order])
)