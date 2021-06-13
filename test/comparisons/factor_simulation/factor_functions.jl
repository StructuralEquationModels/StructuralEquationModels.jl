using CSV, DataFrames, Arrow, sem, ModelingToolkit, 
    LinearAlgebra, SparseArrays, DataFrames, Optim, LineSearches

cd("C:\\Users\\maxim\\.julia\\dev\\sem\\test\\comparisons\\factor_simulation")

config = DataFrame(CSV.File("config_factor.csv"))

function read_files(dir)
    data_paths = readdir(dir; sort = false)
    data = Vector{Any}()
    for i in 1:length(data_paths)
        push!(data, DataFrames.DataFrame(Arrow.Table(dir*"/"*data_paths[i])))
    end
    return data
end

data_vec = read_files("data")

function gen_model(nfact, nitem, data)
    nfact = Int64(nfact)
    nitem = Int64(nitem)
    # observed
    semobserved = SemObsCommon(data = Matrix{Float64}(data))

    ## Model definition
    nobs = nfact*nitem
    nnod = nfact+nobs
    @ModelingToolkit.variables x[1:Int64(2*nobs+(nfact*(nfact+1)/2)-nfact)]

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

    start_val = vcat(
        fill(1.0, nobs),
        fill(0.05, Int64(nfact*(nfact+1)/2)),
        fill(0.05, nobs-nfact)
    )

    # loss
    loss = Loss([SemML(semobserved, [0.0], similar(start_val))])

    # imply
    imply = ImplySymbolic(A, S, F, x, start_val)    

    grad_ml = sem.∇SemML(A, S, F, x, start_val)    

    diff_ana = 
        SemAnalyticDiff(
            LBFGS(), 
            Optim.Options(
                ;f_tol = 1e-10, 
                x_tol = 1.5e-8),
                (grad_ml,))  
                
    return Sem(semobserved, imply, loss, diff_ana)
end
