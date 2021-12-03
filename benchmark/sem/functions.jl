function get_data_paths(config)
    data_paths = []
    for i = 1:nrow(config)
        row = config[i, :]
        miss = replace(row.missingness, "," => ".")
        file = string(
            "n_factors_",
            row.n_factors,
            "_n_items_",
            row.n_items,
            "_missing_",
            miss, 
            ".csv"
            )
        push!(data_paths, file)
    end
    return data_paths
end

function read_files(dir, data_paths)
    data = Vector{Any}()
    for i in 1:length(data_paths)
        push!(data, DataFrame(CSV.read(dir*"/"*data_paths[i], DataFrame; missingstring = "NA")))
    end
    return data
end

function gen_model(nfact, nitem, data, backend)
    nfact = Int64(nfact)
    nitem = Int64(nitem)

    # observed
    semobserved = SemObsMissing(data)

    ## Model definition
    nobs = nfact*nitem
    nnod = nfact+nobs
    npar = 3nobs + nfact-1
    Symbolics.@variables x[1:npar]

    #F
    Ind = collect(1:nobs)
    Jnd = collect(1:nobs)
    V = fill(1,nobs)
    F = sparse(Ind, Jnd, V, nobs, nnod)

    #A
    Ind = collect(1:nobs)
    Jnd = vcat([fill(nobs+i, nitem) for i in 1:nfact]...)
    V = x[1:nobs]
    A = sparse(Ind, Jnd, V, nnod, nnod)
    xind = nobs+1
    for i in nobs+1:nnod-1
        A[i,i+1] = x[xind]
        xind = xind+1
    end

    #S
    Ind = collect(1:nnod)
    Jnd = collect(1:nnod)
    V = [x[nobs+nfact:2nobs+nfact-1]; fill(1.0, nfact)]
    S = sparse(Ind, Jnd, V, nnod, nnod)

    #M
    M = [x[2nobs+nfact:3nobs+nfact-1]...; fill(0.0, nfact)]
    
    start_val = start_simple(Matrix(A), Matrix(S), Matrix(F), x; loadings = 1.0)

    # imply and loss
    semimply = RAMSymbolic(A, S, F, x, start_val; M = M)
    semloss = SemLoss((SemFIML(semobserved, 1.0, similar(start_val)),))
    
    if backend == "Optim.jl"
        semdiff = SemDiffOptim(
            LBFGS(
                ;linesearch = BackTracking(order=3), 
                alphaguess = InitialHagerZhang()
                ),
            Optim.Options(
                ;f_tol = 1e-10,
                x_tol = 1.5e-8)
            )
    elseif backend =="NLopt.jl"
        semdiff = SemDiffNLopt(
            :LD_LBFGS,
            nothing
            )
    end
                
    return Sem(semobserved, semimply, semloss, semdiff)
end

function gen_models(config, data_vec)
    models = []
    for i = 1:nrow(config)
        row = config[i, :]
        model = gen_model(row.n_factors, row.n_items, Matrix(data_vec[i]), row.backend)
        push!(models, model)
    end
    return models
end

function benchmark_models(models)
    benchmarks = []
    for model in models
        model = model
        bm = @benchmark sem_fit(model)
        push!(benchmarks, bm)
    end
    return benchmarks
end

function get_fits(models)
    fits = []
    for model in models
        fit = sem_fit(model)
        push!(fits, fit)
    end
    return fits
end