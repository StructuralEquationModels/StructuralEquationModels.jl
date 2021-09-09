using CSV, DataFrames, Arrow, sem, ModelingToolkit, 
    LinearAlgebra, SparseArrays, DataFrames, Optim, LineSearches,
    BenchmarkTools

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

function gen_model(nfact, nitem, data, start_val)
    nfact = Int64(nfact)
    nitem = Int64(nitem)
    # observed
    #semobserved = SemObsCommon(data = Matrix{Float64}(data))

    semobserved = SemObsMissing(data)

    ## Model definition
    nobs = nfact*nitem
    nnod = nfact+nobs
    @ModelingToolkit.variables x[1:Int64(nobs + nobs)], m[1:nobs]

    #F
    Ind = collect(1:nobs)
    Jnd = collect(1:nobs)
    V = fill(1,nobs)
    F = sparse(Ind, Jnd, V, nobs, nnod)

    #S
    Ind = collect(1:nnod)
    Jnd = collect(1:nnod)
    V = [x[1:nobs]; fill(1.0, nfact)]
    S = sparse(Ind, Jnd, V, nnod, nnod)


    #A
    Ind = collect(1:nobs)
    Jnd = vcat([fill(nobs+i, nitem) for i in 1:nfact]...)
    V = x[(nobs+1):(2*nobs)]
    A = sparse(Ind, Jnd, V, nnod, nnod)

    M = [m..., fill(0.0, nfact)...]

    res_ind = (nobs+1):(2*nobs)
    load_ind = 1:nobs
    n_est = length(start_val)
    mean_ind = (n_est-nobs-nfact+1):(n_est-nfact)

    ind = [collect(res_ind)..., collect(load_ind)..., collect(mean_ind)...]

    start_val = start_val[ind]

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

    grad_fiml = sem.∇SemFIML(semobserved, imply, A, S, F, [x..., m...], start_val; M = M)
    
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

#=     diff_fin = SemFiniteDiff(
        LBFGS(
            ;alphaguess = LineSearches.InitialHagerZhang(),
            linesearch = LineSearches.HagerZhang()
        ), 
        Optim.Options(;
            f_tol = 1e-10, 
            x_tol = 1.5e-8)) =#
                
    return (Sem(semobserved, imply, loss, diff_ana), ind)
end

function gen_models(config, data_vec, start_vec)
    models = []
    for i = 1:nrow(config)
        row = config[i, :]
        model = gen_model(row.nfact_vec, row.nitem_vec, Matrix(data_vec[i]), 
        convert(Vector{Float64}, start_vec[i].est))
        push!(models, model)
    end
    return models
end

function benchmark_models(models)
    benchmarks = []
    for model in models
        bm = @benchmark sem_fit($model)
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

function check_solution(fits, par_vec, par_order; tol = 0.05)
    approx_equal = []
    for (fit, par, ind) in zip(fits, par_vec, par_order)
        is_equ = 
            all(
                abs.(fit.minimizer .- par.est[ind]) .< 
                tol*abs.(par.est[ind]))
        push!(approx_equal, is_equ)
    end
    return approx_equal
end