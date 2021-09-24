############################################################################
### Types

struct ImplySymbolic{
    F <: Any,
    A <: AbstractArray,
    S <: Array{Float64},
    F2 <: Any,
    A2 <: Union{Nothing, AbstractArray}} <: Imply
    imp_fun::F
    imp_cov::A
    start_val::S
    imp_fun_mean::F2
    imp_mean::A2
end

# for forward diff
mutable struct ImplySymbolicForward{
    F <: Any,
    S <: Array{Float64}} <: Imply
    imp_fun::F
    imp_cov
    start_val::S
end

# for reverse diff and others
mutable struct ImplySymbolicAlloc{
    F <: Any,
    S <: Array{Float64}} <: Imply
    imp_fun::F
    imp_cov
    start_val::S
end

struct ImplySymbolicDefinition{
    F <: Any,
    A <: AbstractArray,
    S <: Array{Float64},
    F2 <: Any,
    A2 <: Union{Nothing, AbstractArray},
    I <: Int64,
    P <: Int64,
    R <: AbstractArray,
    D <: AbstractArray} <: ImplyDefinition
imp_fun::F
imp_cov::A # Array of matrices
start_val::S
imp_fun_mean::F2
imp_mean::A2 # Array of matrices of meanstructure
n_obs::I
n_patterns::P
rows::R
data_def::D
end

struct ImplySymbolicWLS{
    F <: Any,
    A <: AbstractArray,
    S <: Array{Float64}} <: Imply
    imp_fun::F
    imp_cov::A
    start_val::S
end

struct ImplySymbolicSWLS{
    F <: Any,
    A <: AbstractArray,
    S <: Array{Float64}} <: Imply
    imp_fun::F
    G::A
    start_val::S
end
############################################################################
### Constructors

function ImplySymbolic(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val;
        M::Spa4 = nothing
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC,
            Spa4 <: Union{Nothing, AbstractArray}
            }

    #Model-implied covmat

    invia = neumann_series(A)

    imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym = ModelingToolkit.simplify.(imp_cov_sym)

    imp_fun =
        eval(ModelingToolkit.build_function(
            imp_cov_sym,
            parameters
        )[2])

    imp_cov = zeros(size(F)[1], size(F)[1])
    #imp_cov = Base.invokelatest(imp_fun_, start_val)
    #Model implied mean
    if !isnothing(M)
        imp_mean_sym = F*invia*M
        imp_mean_sym = Array(imp_mean_sym)
        imp_mean_sym = ModelingToolkit.simplify.(imp_mean_sym)

        imp_fun_mean =
            eval(ModelingToolkit.build_function(
                imp_mean_sym,
                parameters
            )[2])

        imp_mean = zeros(size(F)[1])
    else
        imp_fun_mean = nothing
        imp_mean = nothing
    end

    return ImplySymbolic(
        imp_fun,
        imp_cov,
        copy(start_val),
        imp_fun_mean,
        imp_mean
    )
end

function ImplySymbolicWLS(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC,
            Spa4 <: Union{Nothing, AbstractArray}
            }

    #Model-implied covmat

    invia = neumann_series(A)

    imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym = ModelingToolkit.simplify.(imp_cov_sym)
    imp_cov_sym = LowerTriangular(imp_cov_sym)
    imp_cov_sym = sparse(imp_cov_sym)
    imp_cov_sym = imp_cov_sym.nzval

    imp_fun =
        eval(ModelingToolkit.build_function(
            imp_cov_sym,
            parameters
        )[2])

    imp_cov = zeros(Int64(0.5*size(F)[1]*(size(F)[1] + 1)))

    return ImplySymbolicWLS(
        imp_fun,
        imp_cov,
        copy(start_val)
    )
end

function ImplySymbolicSWLS(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC,
            Spa4 <: Union{Nothing, AbstractArray}
            }

    nobs, ntot = size(F)
    nlat = ntot - nobs

    # Selection matrices
    ind = findall(tril(trues(nobs, nobs)))
    Es = []
    for i in ind
        E = zeros(Bool, nobs, nobs)
        E[i] = true
        E = vec(E)
        push!(Es, E)
    end
    L = hcat(Es...)
    K = L*inv(transpose(L)*L)
    K = sparse(K)

    ind = findall(!iszero, S)
    ind = filter(x -> (x[1] >= x[2]), ind)
    Es = []
    for i in ind
        E = zeros(Bool, nobs+nlat, nobs+nlat)
        E[i] = true
        E = vec(E)
        push!(Es, E)
    end
    L_Ω = hcat(Es...)
    L_Ω = sparse(L_Ω)

    #Model-implied covmat
    A = neumann_series(A)

    G = transpose(K)*(kron(F, F)*kron(A, A))*L_Ω
    G = Array(G)
    G = ModelingToolkit.simplify.(G)

    imp_fun =
        eval(ModelingToolkit.build_function(
            G,
            parameters
        )[2])

    G = zeros(size(G))

    return ImplySymbolicSWLS(
        imp_fun,
        G,
        copy(start_val)
    )
end

function ImplySymbolicAlloc(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC
            }

    invia = neumann_series(A)

    imp_cov_sym = F*invia*S*invia'*F'

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym .= ModelingToolkit.simplify.(imp_cov_sym)

    imp_fun =
        eval(ModelingToolkit.build_function(
            imp_cov_sym,
            parameters
        )[1])

    imp_cov = rand(size(F)[1], size(F)[1])
    #imp_cov = Base.invokelatest(imp_fun_, start_val)
    return ImplySymbolicAlloc(imp_fun, imp_cov, copy(start_val))
end

function ImplySymbolicForward(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC
            }

    invia = neumann_series(A)

    imp_cov_sym = F*invia*S*invia'*F'

    imp_cov_sym = Array(imp_cov_sym)
    imp_cov_sym .= ModelingToolkit.simplify.(imp_cov_sym)

    #imp_cov_sym = DiffEqBase.dualcache(imp_cov_sym)

    imp_fun =
        eval(ModelingToolkit.build_function(
            imp_cov_sym,
            parameters
        )[1])

    imp_cov = DiffEqBase.dualcache(zeros(size(F)[1], size(F)[1]))
    #imp_cov = Base.invokelatest(imp_fun_, start_val)
    return ImplySymbolicForward(imp_fun, imp_cov, copy(start_val))
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
    n_def_vars = Float64(size(data_def, 2))
    
    patterns = [data_def[i, :] for i = 1:n_obs]
    remember = Vector{Vector{Float64}}()
    rows = [Vector{Int64}(undef, 0) for i = 1:n_obs]
    
    for i = 1:size(patterns, 1)
        unknown = true
        for j = 1:size(remember, 1)
            if patterns[i] == remember[j]
                push!(rows[j], i)
                unknown = false
            end
        end
        if unknown
            push!(remember, patterns[i])
            push!(rows[size(remember, 1)], i)
        end
    end
    
    rows = rows[1:length(remember)]
    n_patterns = size(rows, 1)
    
    pattern_n_obs = size.(rows, 1)
    
    #############################################
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
    
    imp_cov = [zeros(size(F)[1], size(F)[1]) for i = 1:n_patterns]
    
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
    
    imp_mean = [zeros(size(F)[1]) for i = 1:n_patterns]
    
    data_def = remember
    
    return ImplySymbolicDefinition(
        imp_fun,
        imp_cov,
        copy(start_val),
        imp_fun_mean,
        imp_mean,
        n_obs,
        n_patterns,
        rows,
        data_def
    )
end
    



############################################################################
### loss functions

function (imply::ImplySymbolic)(parameters, model)
    imply.imp_fun(imply.imp_cov, parameters)
    if !isnothing(imply.imp_mean)
        imply.imp_fun_mean(imply.imp_mean, parameters)
    end
end

function (imply::ImplySymbolicWLS)(parameters, model)
    imply.imp_fun(imply.imp_cov, parameters)
end

function (imply::ImplySymbolicSWLS)(parameters, model)
    imply.imp_fun(imply.G, parameters)
end

function (imply::ImplySymbolicAlloc)(parameters, model)
    imply.imp_cov = imply.imp_fun(parameters)
end

function (imply::ImplySymbolicForward)(parameters, model)
    imply.imp_cov = DiffEqBase.get_tmp(imply.imp_cov, parameters)
    imply.imp_fun(imply.imp_cov, parameters)
end

function (imply::ImplySymbolicDefinition)(parameters, model)
    for i = 1:imply.n_patterns
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