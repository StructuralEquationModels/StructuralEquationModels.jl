using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff

## Observed Data
growth_dat = Feather.read("test/comparisons/growth_dat.feather")
growth_par = Feather.read("test/comparisons/growth_par.feather")

definition_dat = Feather.read("test/comparisons/definition_dat.feather")
definition_par = Feather.read("test/comparisons/definition_par.feather")

semobserved = SemObsCommon(data = Matrix(growth_dat); meanstructure = true)

diff_fin = SemFiniteDiff(BFGS(), Optim.Options())

## Model definition
@ModelingToolkit.variables x[1:7], m[1:2]

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

A =[0  0  0  0  1  0
    0  0  0  0  1  1.0  
    0  0  0  0  1  2  
    0  0  0  0  1  3 
    0  0  0  0  0  0 
    0  0  0  0  0  0]

M = [0.0, 0, 0, 0, m[1:2]...]

#M = sparse(M)

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)

start_val = [1.0, 1, 1, 1, 0.05, 0.05, 0.0, 1.0, 1.0]

loss = Loss([SemML(semobserved, [0.0], similar(start_val))])

imply = ImplySymbolic(A, S, F, [x..., m[1:2]...], start_val; M = M)

#@benchmark imply_sim(start_val)

model_fin = Sem(semobserved, imply, loss, diff_fin)

solution_fin = sem_fit(model_fin)

par_order = [collect(9:15); 20; 21]

all(
    abs.(solution_fin.minimizer .- growth_par.est[par_order]
        ) .< 0.05*abs.(growth_par.est[par_order]))

#start_lav = three_path_start.start[par_order]

###
## Model definition
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
   

#abstract type Imply end

struct ImplySymbolicDefinition{
    F <: Any,
    A <: AbstractArray,
    S <: Array{Float64},
    F2 <: Any,
    A2 <: Union{Nothing, AbstractArray},
    I <: Int64,
    D <: AbstractArray} <: Imply

    imp_fun::F
    imp_cov::A # Array of matrices
    start_val::S
    imp_fun_mean::F2
    imp_mean::A2 # Array of matrices of meanstructure
    n_obs::I
    data_def::D
end

function ImplySymbolicDefinition(
    A::Spa1,
    S::Spa2,
    F::Spa3,
    M::Spa4,
    parameters,
    def_vars,
    start_val,
    data_def
        ) where {
        Spa1 <: SparseMatrixCSC,
        Spa2 <: SparseMatrixCSC,
        Spa3 <: SparseMatrixCSC,
        Spa4 <: Union{Nothing, AbstractArray}
        }

    n_obs = size(data_def, 1)
        
    #Model-implied covmat

    invia = sem.neumann_series(A)

    imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym = ModelingToolkit.simplify.(imp_cov_sym)

    imp_fun =
        eval(ModelingToolkit.build_function(
            imp_cov_sym,
            parameters,
            def_vars
        )[2])

    imp_cov = [zeros(size(F)[1], size(F)[1]) for i = 1:n_obs]

    #Model implied mean
    imp_mean_sym = F*invia*M
    imp_mean_sym = Array(imp_mean_sym)
    imp_mean_sym = ModelingToolkit.simplify.(imp_mean_sym)

    imp_fun_mean =
        eval(ModelingToolkit.build_function(
            imp_mean_sym,
            parameters,
            def_vars
        )[2])

    imp_mean = [zeros(size(F)[1]) for i = 1:n_obs]

    data_def = [data_def[i, :] for i = 1:size(data_def, 1)]

    return ImplySymbolicDefinition(
        imp_fun,
        imp_cov,
        copy(start_val),
        imp_fun_mean,
        imp_mean,
        n_obs,
        data_def
    )
end

function(imply::ImplySymbolicDefinition)(parameters)
    for i = 1:imply.n_obs
        let (cov, 
            mean, 
            def_vars) = 
                (imply.imp_cov[i], 
                imply.imp_mean[i], 
                imply.data_def[i])
            imply.imp_fun(cov, parameters, def_vars)
            imply.imp_fun_mean(mean, parameters, def_vars)
        end
    end
end


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


@benchmark imply($start_val)

struct SemDefinition{ #################### call it per person or sth????
        INV <: AbstractArray,
        C <: AbstractArray,
        L <: AbstractArray,
        M <: AbstractArray,
        I <: AbstractArray,
        T <: AbstractArray,
        DP <: AbstractArray,
        U,
        V} <: LossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats
    meandiff::M
    imp_inv::I
    mult::T
    data_perperson::DP
    objective::U
    grad::V
end

function SemDefinition(observed::O where {O <: SemObs}, objective, grad) 
    n_obs = Int64(observed.n_obs)
    n_man = Int64(observed.n_man)

    inverses = [zeros(n_man, n_man) for i = 1:n_obs]
    choleskys = Array{Cholesky{Float64,Array{Float64,2}},1}(undef, n_obs)
    logdets = zeros(n_obs)

    meandiff = [zeros(n_man) for i = 1:n_obs]

    imp_inv = zeros(size(observed.data, 2), size(observed.data, 2))
    mult = similar.(inverses)

    data_perperson = [observed.data[i, :] for i = 1:n_obs]

    return SemDefinition(
    inverses,
    choleskys,
    logdets,
    meandiff,
    imp_inv,
    mult,
    data_perperson,
    copy(objective),
    copy(grad)
    )
end

function (semdef::SemDefinition)(par, model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemFiniteDiff}

    if isnothing(model.imply.imp_mean) 
        error("A model implied meanstructure is needed for Definition Variables")
    end

    for i = 1:size(semdef.choleskys, 1)
        semdef.choleskys[i] = 
            cholesky!(Hermitian(model.imply.imp_cov[i]); check = false)
    end

    if any(.!isposdef.(semdef.choleskys))
        return Inf
    end

    semdef.logdets .= logdet.(semdef.choleskys)

    for i = 1:size(semdef.inverses, 1)
        semdef.inverses[i] .= LinearAlgebra.inv!(semdef.choleskys[i])
    end

    F = zero(eltype(par))

    for i = 1:Int64(model.observed.n_obs)
        let (imp_mean, meandiff, inverse, data, logdet) =
            (model.imply.imp_mean[i],
            semdef.meandiff[i],
            semdef.inverses[i],
            semdef.data_perperson[i],
            semdef.logdets[i])

            F += F_one_person(imp_mean, meandiff, inverse, data, logdet)

        end
    end
    return F
end

loss = Loss([SemDefinition(semobserved, 0.0, 0.0)])

model_fin = Sem(semobserved, imply, loss, diff_fin)

model_fin(start_val)

solution = sem_fit(model_fin)

par_order = [collect(1:5); 7; 6; 8; 9]

all(
    abs.(solution.minimizer .- definition_par.Estimate[par_order]
        ) .< 0.05*abs.(definition_par.Estimate[par_order]))