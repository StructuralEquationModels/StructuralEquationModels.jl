function (semmg)(par)
    for i in 1:length(semmg.vec)
        F += semmg.vec[i](par)
    end
end

function (semmg)(par)
    for i in 1:length(semmg.imply_vec)
        semmg.imply_vec[i](par, semmg)
    end
    for i in 1:length(semmg.loss_vec)
        F += semmg.loss_vec[i](par, semmg)
    end
end

function (semmg)(par)
    semmg.imply_column(par, semmg)
    F = semmg.loss_column(par, semmg)
end

abstract type loss_column end

struct ML <: loss_column
    vec::Vector{loss}
end

(ML)(par)

struct ML_superparallel <: loss_column

end

struct WLS_superparallel <: loss_column

end

function (WLS_superparallel)(par, model)
    for i in 1:length(imply_column)
        A[i] = model.imply_column[i].imp_cov
    end
    do A end 
end

sem_fit(model)
sem_fit_superparallel_ML(model)
sem_fit_superparallel_WLS(model)
ML_superparallel(model, par)

function (semdef)(par)
    for i in 1:length(semdef.vec)
        F += semdef.vec[i](par, semdef.defvars[i])
    end
end


function (semfiml)(par)
    for i in 1:length(semfiml.vec)
        F += semfiml.vec[i](par, semfiml.filter[i])
    end
end

function (semfiml)(par, defvars)
    imply(par)
    delete(semfiml.imply)
    for loss(par, semfiml) end

    for i in 1:length(semfiml.vec)
        F += semfiml.vec[i](par, semfiml.filter[i], defvars)
    end
end

function (semel)(par)
    semel.imply(par, semel)
    F = semel.loss(par, semel)
    return F
end

function (semel)(par, filter)
    semel.imply(par, semel)
    F = semel.loss(par, semel, filter)
    return F
end

function (sematom)(par, defvars)
    semel.imply(par, semel, defvars)
    F = semel.loss(par, sematom)
    return F
end

function (semel)(par, defvars, filter)
    semel.imply(par, semel, defvars)
    F = semel.loss(par, semel, filter)
    return F
end

test

Threads.nthreads()

A = rand(100,100)

using BenchmarkTools, LinearAlgebra

@benchmark inv(A) 

BLAS.get_config()

using MKL

@benchmark inv(A)

BLAS.get_config()