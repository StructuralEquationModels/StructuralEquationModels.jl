using Symbolics, SparseArrays, StructuralEquationModels, Optim, LineSearches
import StructuralEquationModels as SEM
include("test_helpers.jl")

############################################################################
### observed data
############################################################################

using LinearAlgebra
dat = DataFrame(CSV.File("examples/data/data_dem.csv"))
par_ml = DataFrame(CSV.File("examples/data/par_dem_ml.csv"))
par_order = []

############################################################################
### define models
############################################################################

nfact = 5
nitem = 40

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
V = [x...][1:nobs]
A = sparse(Ind, Jnd, V, nnod, nnod)
xind = nobs+1
for i in nobs+1:nnod-1
    A[i,i+1] = x[xind]
    xind = xind+1
end

#S
Ind = collect(1:nnod)
Jnd = collect(1:nnod)
V = [[x...][nobs+nfact:2nobs+nfact-1]; fill(1.0, nfact)]
S = sparse(Ind, Jnd, V, nnod, nnod)

#M
M = [[x...][2nobs+nfact:3nobs+nfact-1]...; fill(0.0, nfact)]

start_val = start_simple(Matrix(A), Matrix(S), Matrix(F), x; loadings = 1.0)

# imply and loss
semimply = RAMSymbolic(A, S, F, x, start_val; M = M)
semobserved = SemObsCommon(data = dat, specification = lol)
semloss = SemLoss((SemFIML(semobserved, 1.0, similar(start_val)),))

semdiff = SemDiffOptim(
    LBFGS(
        ;linesearch = BackTracking(order=3), 
        alphaguess = InitialHagerZhang()
        ),
    Optim.Options(
        ;f_tol = 1e-10,
        x_tol = 1.5e-8)
    )
            
model_ml = Sem(SpecEmpty(), semobserved, semimply, semloss, semdiff)



############################################################################
### test gradients
############################################################################

@testset "ml_gradients_big_sem" begin
    @test test_gradient(model_ml, start_val)
end

############################################################################
### test solution
############################################################################

@testset "ml_solution_big_sem" begin
    solution_ml = sem_fit(model_ml)
    @test SEM.compare_estimates(par_ml.est[par_order], solution_ml.solution, 0.01)
end