using sem, Arrow, ModelingToolkit, LinearAlgebra, SparseArrays, DataFrames, Optim, LineSearches

cd("./test")
## Observed Data
dat = DataFrame(Arrow.Table("comparisons/reg_1.arrow"))
par = DataFrame(Arrow.Table("comparisons/reg_1_par.arrow"))
start_lav = DataFrame(Arrow.Table("comparisons/reg_1_start.arrow"))

# observed
semobserved = SemObsCommon(data = Matrix{Float64}(dat))

## Model definition
@ModelingToolkit.variables x[1:243]

#F
Ind = collect(1:25)
Jnd = collect(1:25)
V = fill(1,25)
F = sparse(Ind, Jnd, V, 25, 46)

#A
Ind = [collect(2:25); 5; collect(6:25); fill(27, 19); 1; 4]
Jnd = [fill(26, 24); 27; collect(27:46); collect(28:46); 26; 27]
V =[x[1:64]; 1.0; 1.0]
A = sparse(Ind, Jnd, V, 46, 46)

#S
Ind = [collect(1:6); 26; 27; collect(28:46)]
Jnd = [collect(1:6); 26; 27; collect(28:46)]
V = [x[65:72]; fill(1.0, 19)]
S = sparse(Ind, Jnd, V, 46, 46)
xind = 73
for i = 28:46
    for j = (i+1):46
       S[i,j] = x[xind]
       S[j,i] = x[xind]
       global xind += 1
    end
end

par_order = [
    collect(2:6); 
    collect(11:29); 
    8;9; 
    collect(49:67);
    collect(106:124); 
    collect(125:130);
    131; 132;
    collect(133:303)]

start_val = Vector{Float64}(start_lav.est[par_order])   

# loss
loss = Loss([SemML(semobserved, [0.0], similar(start_val))])

# imply
imply = ImplySymbolic(A, S, F, x, start_val)

# diff
grad_ml = sem.∇SemML(A, S, F, x, start_val)       

diff_ana = 
    SemAnalyticDiff(
        LBFGS(
            m = 50,
            alphaguess = InitialHagerZhang(), 
            linesearch = HagerZhang()), 
        Optim.Options(
            ;f_tol = 1e-10, 
            x_tol = 1.5e-8),
            (grad_ml,))  
            
model_ana = Sem(semobserved, imply, loss, diff_ana)

using BenchmarkTools

@benchmark solution_ana = sem_fit(model_ana)

all(abs.(solution_ana.minimizer .- par.est[par_order]) .< 0.05*abs.(par.est[par_order]))