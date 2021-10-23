using FiniteDiff

abstract type AbstractSem end

## loss
abstract type SemLossFunction end

struct SemLoss{F <: Tuple}
    functions::F
end

## Diff
abstract type SemDiff end

## Obs
abstract type SemObs end

## Imply
abstract type SemImply end

struct Sem{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff} <: AbstractSem
    observed::O
    imply::I 
    loss::L 
    diff::D
end

function (model::Sem)(par, F, G, H, weight = nothing)
    model.imply(par, F, G, H, model)
    F = model.loss(par, F, G, H, model, weight)
    if !isnothing(weight) F = weight*F end
    return F
end

function (loss::SemLoss)(par, F, G, H, model)
    if !isnothing(F)
        F = zero(eltype(par))
        for lossfun in loss.functions
            F += lossfun(par, F, G, H, model)
        end
        return F
    end
    for lossfun in loss.functions lossfun(par, F, G, H, model) end
end

function (loss::SemLoss)(par, F, G, H, model, weight)
    if !isnothing(F)
        F = zero(eltype(par))
        for lossfun in loss.functions
            F += lossfun(par, F, G, H, model)
        end
        return F
    end
    for lossfun in loss.functions lossfun(par, F, G, H, model) end
end

######################## example ############################

struct myobs <: SemObs end

obsinst = myobs()

struct mydiff <: SemDiff end

diffinst = mydiff()

struct myimply <: SemImply
    Σ
end

struct myml <: SemLossFunction
    Σ
end

struct myhell <: SemLossFunction
    Σ
end

mlinst = myml([0.0])
hellinst = myhell([0.0])
implyinst = myimply([0.0])

modelinst = Sem(obsinst, implyinst, SemLoss((mlinst,hellinst)), diffinst)

function (imply::myimply)(par, F, G, H, model)
    imply.Σ[1] = par^2
end

function (lossfun::myml)(par, F, G, H, model, weight = nothing)
    # do common computations here
    if !isnothing(G)
        if isnothing(weight) G[1] += 4*par else G[1] += weight*4*par end
    end
    # if isnothing(H) end
    if !isnothing(F)
        F = 2*model.imply.Σ[1]
        if !isnothing(weight) F = weight*F end
        return F
    end
end

function (lossfun::myhell)(par, F, G, H, model)
    if !isnothing(G)
        G .+= FiniteDiff.finite_difference_gradient(par -> lossfun(par, model), [par])
    end
    #if isnothing(H) end
    if !isnothing(F)
        F = model.imply.Σ[1]^2
        return F
    end
end

function (lossfun::myhell)(par, model)
    lossfun.Σ[1] = par[1]^2
    return lossfun.Σ[1]^2
end

par = 2.0

grad = [0.0]

2*par^2 + par^4

4*par + 4*par^3

modelinst(par, 0.0, grad, nothing)

grad

using Optim, BenchmarkTools

@benchmark sol_fin = optimize(par -> modelinst(par[1], 0.0, nothing, nothing), [par], LBFGS())

sol_fin.minimizer

@benchmark sol_grad = optimize(Optim.only_fg!((F, G, par) -> modelinst(par[1], F, G, nothing)), [par], LBFGS())

sol_grad.minimizer


################################# system 2 ######################################

################## specify sum in ensemble ##############


struct myobs <: SemObs end

obsinst = myobs()

struct mydiff <: SemDiff end

diffinst = mydiff()

struct myimply <: SemImply
    Σ
end

struct myml <: SemLossFunction end

struct myhell <: SemLossFunction
    Σ
end

mlinst = myml([0.0])
hellinst = myhell([0.0])
implyinst = myimply([0.0])

modelinst = Sem(obsinst, implyinst, SemLoss((mlinst,hellinst)), diffinst)

function (imply::myimply)(par, F, G, H, model)
    imply.Σ[1] = par^2
end

function (lossfun::myml)(par, F, G, H, model, weight = nothing)
    # do common computations here
    if !isnothing(G)
        if isnothing(weight) G[1] += 4*par else G[1] += weight*4*par end
    end
    # if isnothing(H) end
    if !isnothing(F)
        F = 2*model.imply.Σ[1]
        if !isnothing(weight) F = weight*F end
        return F
    end
end

function (lossfun::myhell)(par, F, G, H, model, weight = nothing)
    if !isnothing(G)
        if !isnothing() G .+= FiniteDiff.finite_difference_gradient(par -> lossfun(par, model), [par]) end
    end
    #if isnothing(H) end
    if !isnothing(F)
        F = model.imply.Σ[1]^2
        return F
    end
end

function (lossfun::myhell)(par, model)
    lossfun.Σ[1] = par[1]^2
    return lossfun.Σ[1]^2
end

par = 2.0

grad = [0.0]

2*par^2 + par^4

4*par + 4*par^3

modelinst(par, 0.0, grad, nothing)

grad

using Optim, BenchmarkTools

@benchmark sol_fin = optimize(par -> modelinst(par[1], 0.0, nothing, nothing), [par], LBFGS())

sol_fin.minimizer

@benchmark sol_grad = optimize(Optim.only_fg!((F, G, par) -> modelinst(par[1], F, G, nothing)), [par], LBFGS())

sol_grad.minimizer

function myf1(a)
    if isnothing(a) error("there is a problem") end
end

function myf2(a)
    return a
end

using ForwardDiff, BenchmarkTools

a = 1.0

@benchmark myf1($a)