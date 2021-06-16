using CSV, DataFrames, Arrow, sem, ModelingToolkit, 
    LinearAlgebra, SparseArrays, DataFrames, Optim, LineSearches

cd("C:\\Users\\maxim\\.julia\\dev\\sem\\test\\comparisons\\factor_simulation")

config = DataFrame(CSV.File("config_factor.csv"))



function get_data_paths(config)
    data_paths = []
    for i = 1:nrow(config)
        row = config[i, :]
        file = string("nfact_", row.nfact_vec, "_nitem_", row.nitem_vec, ".arrow")
        push!(data_paths, file)
    end
    return data_paths
end

function read_files(dir, data_paths)
    data = Vector{Any}()
    for i in 1:length(data_paths)
        push!(data, DataFrames.DataFrame(Arrow.Table(dir*"/"*data_paths[i])))
    end
    return data
end

data_vec = read_files("data", get_data_paths(config))

function gen_model(nfact, nitem, data)
    nfact = Int64(nfact)
    nitem = Int64(nitem)
    # observed
    #semobserved = SemObsCommon(data = Matrix{Float64}(data))

    semobserved = SemObsMissing(data)

    ## Model definition
    nobs = nfact*nitem
    nnod = nfact+nobs
    @ModelingToolkit.variables x[1:Int64(2*nobs+(nfact*(nfact+1)/2)-nfact)], m[1:nobs]

    #F
    Ind = collect(1:nobs)
    Jnd = collect(1:nobs)
    V = fill(1,nobs)
    F = sparse(Ind, Jnd, V, nobs, nnod)

    #S
    Ind = collect(1:nnod)
    Jnd = collect(1:nnod)
    V = x[1:nnod]
    S = sparse(Ind, Jnd, V, nnod, nnod)
    xind = nnod + 1
    for i = nobs+1:nnod
        for j = (i+1):nnod
            S[i,j] = x[xind]
            S[j,i] = x[xind]
            xind += 1
        end
    end

    #A
    Ind = collect(0:nfact-1).*nitem .+ 1
    Jnd = collect(nobs+1:nnod)
    V = fill(Num(1), nfact)
    A = sparse(Ind, Jnd, V, nnod, nnod)

    for i = 1:nfact
        for j = ((i-1)*nitem+2):((i-1)*nitem+nitem)
            A[j, i+nobs] = x[xind]
            xind += 1
        end
    end

    M = [m..., fill(0.0, nfact)...]

    start_val = vcat(
        fill(1.0, nobs),
        fill(0.05, Int64(nfact*(nfact+1)/2)),
        fill(0.05, nobs-nfact),
        fill(0.0, nobs)
    )

    # imply
    imply = ImplySymbolic(A, S, F, [x..., m...], start_val; M = M)   

    # loss
    #loss = Loss([SemML(semobserved, [0.0], similar(start_val))])
    loss = Loss([SemFIML(semobserved, imply, 0.0, 0.0)])

    #grad_ml = sem.∇SemML(A, S, F, x, start_val)    

#=     diff_ana = 
        SemAnalyticDiff(
            LBFGS(), 
            Optim.Options(
                ;f_tol = 1e-10, 
                x_tol = 1.5e-8),
                (grad_ml,))   =#

    diff_fin = SemFiniteDiff(
        LBFGS(), 
        Optim.Options(;
            f_tol = 1e-10, 
            x_tol = 1.5e-8))
                
    return Sem(semobserved, imply, loss, diff_fin)
end

testsem = gen_model(3, 5, Matrix(data_vec[1]))

solution = sem_fit(testsem)