using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, ReverseDiff, Zygote, ForwardDiff,
    BenchmarkTools, Optim

## Observed Data
three_path_dat = Feather.read("test/comparisons/three_path_dat.feather")
three_path_par = Feather.read("test/comparisons/three_path_par.feather")

semobserved = SemObsCommon(data = Matrix(three_path_dat))

loss = Loss([SemML(semobserved)])

diff = SemForwardDiff(LBFGS(), Optim.Options())

diff2 = SemFiniteDiff(LBFGS(), Optim.Options())

## Model definition
@variables x[1:31]

S =[x[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
    0     x[2]  0     0     0     0     0     0     0     0     0     0     0     0
    0     0     x[3]  0     0     0     0     0     0     0     0     0     0     0
    0     0     0     x[4]  0     0     0     x[15] 0     0     0     0     0     0
    0     0     0     0     x[5]  0     x[16] 0     x[17] 0     0     0     0     0
    0     0     0     0     0     x[6]  0     0     0     x[18] 0     0     0     0
    0     0     0     0     x[16] 0     x[7]  0     0     0     x[19] 0     0     0
    0     0     0     x[15] 0     0     0     x[8]  0     0     0     0     0     0
    0     0     0     0     x[17] 0     0     0     x[9]  0     x[20] 0     0     0
    0     0     0     0     0     x[18] 0     0     0     x[10] 0     0     0     0
    0     0     0     0     0     0     x[19] 0     x[20] 0     x[11] 0     0     0
    0     0     0     0     0     0     0     0     0     0     0     x[12] 0     0
    0     0     0     0     0     0     0     0     0     0     0     0     x[13] 0
    0     0     0     0     0     0     0     0     0     0     0     0     0     x[14]]

F =[1.0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 0 0]

A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[21] 0     0
    0  0  0  0  0  0  0  0  0  0  0     x[22] 0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0  0  0     0     x[23] 0
    0  0  0  0  0  0  0  0  0  0  0     0     x[24] 0
    0  0  0  0  0  0  0  0  0  0  0     0     x[25] 0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[26]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[27]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[28]
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[29] 0     0
    0  0  0  0  0  0  0  0  0  0  0     x[30] x[31] 0]


S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)

start_val = vcat(
    vec(var(Matrix(three_path_dat), dims = 1))./2,
    fill(0.05, 3),
    fill(0.0, 6),
    fill(1.0, 8),
    fill(0, 3)
    )



imply = sem.ImplySymbolicAlloc(A, S, F, x, start_val)
imply2 = ImplySymbolic(A, S, F, x, start_val)

model = Sem(semobserved, imply, loss, diff)
model2 = Sem(semobserved, imply2, loss, diff2)

model(start_val)

model

ForwardDiff.gradient(model, start_val)

@btime sol1 = optimize(model2, start_val, LBFGS())
@btime sol2 = optimize(model, start_val, LBFGS(), autodiff = :forward)

sem_fit(model2)

pars = [1.0 0]
pars2 = fill(1.0,4)

Zygote.@nograd isposdef

A = DiffEqBase.dualcache(zeros(2,2))

function myf(pars, A)
    A = DiffEqBase.get_tmp(A, pars)
    A[1] = pars[1]
    A[2] = pars[2]
    A[3] = pars[2]
    A[4] = pars[1]

    if !isposdef(A)
        return Inf
    else
        F = logdet(A) + tr(inv(A))
        return F
    end
end

function myf(pars)
    A = [pars[1] pars[2]
        pars[2] pars[1]]

    if !isposdef(A)
        return Inf
    else
        F = logdet(A) + tr(inv(A))
        return F
    end
end

Zygote.gradient(myf, pars, A)
ForwardDiff.gradient(par -> myf(par, A), pars)


@benchmark sol1 = optimize(par -> myf(par, A), pars; autodiff = :forward)
@benchmark sol2 = optimize(par -> myf(par), pars; autodiff = :forward)


sol1.minimizer

sol2 = optimize(myf,
    par -> Zygote.gradient(myf, par)[1], pars;
    inplace = false)

sol2.minimizer

sol3 = optimize(par -> myf(par), pars, autodiff = :forward)
sol3.minimizer

## ForwardDiff gradients with build functions
@variables x[1:4]

A = [x[1] x[2]
    x[3] x[4]]

A = UpperTriangular(A)

myf2 = eval.(ModelingToolkit.build_function(A, x))[2]

mat = rand(2,2)

mat = UpperTriangular(mat)

myf2(mat, fill(1.0, 4))

myf3(x) = x[1]^2

ForwardDiff.gradient(myf2, [1.0])
