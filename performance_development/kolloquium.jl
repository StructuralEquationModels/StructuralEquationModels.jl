using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff, BenchmarkTools

## Model definition
@ModelingToolkit.variables θ[1:31]

S =[θ[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
    0     θ[2]  0     0     0     0     0     0     0     0     0     0     0     0
    0     0     θ[3]  0     0     0     0     0     0     0     0     0     0     0
    0     0     0     θ[4]  0     0     0     θ[15] 0     0     0     0     0     0
    0     0     0     0     θ[5]  0     θ[16] 0     θ[17] 0     0     0     0     0
    0     0     0     0     0     θ[6]  0     0     0     θ[18] 0     0     0     0
    0     0     0     0     θ[16] 0     θ[7]  0     0     0     θ[19] 0     0     0
    0     0     0     θ[15] 0     0     0     θ[8]  0     0     0     0     0     0
    0     0     0     0     θ[17] 0     0     0     θ[9]  0     θ[20] 0     0     0
    0     0     0     0     0     θ[18] 0     0     0     θ[10] 0     0     0     0
    0     0     0     0     0     0     θ[19] 0     θ[20] 0     θ[11] 0     0     0
    0     0     0     0     0     0     0     0     0     0     0     θ[12] 0     0
    0     0     0     0     0     0     0     0     0     0     0     0     θ[13] 0
    0     0     0     0     0     0     0     0     0     0     0     0     0     θ[14]]

#S = sparse(S)    
LS = latexify(S)
LS = String(LS)    
LS = replace(LS, "ModelingToolkit.Constant(" => "")
LS = replace(LS, ")" => "")

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
    0  0  0  0  0  0  0  0  0  0  0     θ[21] 0     0
    0  0  0  0  0  0  0  0  0  0  0     θ[22] 0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0  0  0     0     θ[23] 0
    0  0  0  0  0  0  0  0  0  0  0     0     θ[24] 0
    0  0  0  0  0  0  0  0  0  0  0     0     θ[25] 0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     θ[26]
    0  0  0  0  0  0  0  0  0  0  0     0     0     θ[27]
    0  0  0  0  0  0  0  0  0  0  0     0     0     θ[28]
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     θ[29] 0     0
    0  0  0  0  0  0  0  0  0  0  0     θ[30] θ[31] 0]

LS = latexify(A)
LS = String(LS)    
LS = replace(LS, "ModelingToolkit.Constant(" => "")
LS = replace(LS, ")" => "")
print(LS)

S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)

invia = sem.neumann_series(A)
    
imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)

imp_cov_sym = Array(imp_cov_sym)
imp_cov_sym = ModelingToolkit.simplify.(imp_cov_sym)

LI = latexify(imp_cov_sym)
LI = String(LI)
LI = replace(LI, "θ" => "\\theta")

parameters = θ

ModelingToolkit.build_function(
        imp_cov_sym,
        θ
    )[2]

imp_fun =
    eval(ModelingToolkit.build_function(
        imp_cov_sym,
        θ
    )[2])

imp_fun()

start_val = vcat(
    fill(1.0, 11),
    fill(0.05, 3),
    fill(0.0, 6),
    fill(1.0, 8),
    fill(0, 3)
    )

pre = zeros(11, 11)

@benchmark imp_fun(pre, start_val)

A_real =    [0  0  0  0  0  0  0  0  0  0  0    1.0     0     0
            0  0  0  0  0  0  0  0  0  0  0     1     0     0
            0  0  0  0  0  0  0  0  0  0  0     1     0     0
            0  0  0  0  0  0  0  0  0  0  0     0     1     0
            0  0  0  0  0  0  0  0  0  0  0     0     1     0
            0  0  0  0  0  0  0  0  0  0  0     0     1     0
            0  0  0  0  0  0  0  0  0  0  0     0     1     0
            0  0  0  0  0  0  0  0  0  0  0     0     0     1
            0  0  0  0  0  0  0  0  0  0  0     0     0     1
            0  0  0  0  0  0  0  0  0  0  0     0     0     1
            0  0  0  0  0  0  0  0  0  0  0     0     0     1
            0  0  0  0  0  0  0  0  0  0  0     0     0     0
            0  0  0  0  0  0  0  0  0  0  0     0     0     0
            0  0  0  0  0  0  0  0  0  0  0     0     0     0]

pre2 = zeros(14,14)
pre3 = copy(pre2)
pre4 = copy(pre2)

A_real = I-A_real

S =[1  0     0     0     0     0     0     0     0     0     0     0     0     0
    0     1  0     0     0     0     0     0     0     0     0     0     0     0
    0     0     1  0     0     0     0     0     0     0     0     0     0     0
    0     0     0     1  0     0     0     0 0     0     0     0     0     0
    0     0     0     0     1  0     0 0     0 0     0     0     0     0
    0     0     0     0     0     1  0     0     0     0 0     0     0     0
    0     0     0     0     0 0     1  0     0     0     0 0     0     0
    0     0     0     0 0     0     0     1  0     0     0     0     0     0
    0     0     0     0     0 0     0     0     1  0     0 0     0     0
    0     0     0     0     0     0 0     0     0     1 0     0     0     0
    0     0     0     0     0     0     0 0     0 0     1 0     0     0
    0     0     0     0     0     0     0     0     0     0     0     0.05 0     0
    0     0     0     0     0     0     0     0     0     0     0     0     0.05 0
    0     0     0     0     0     0     0     0     0     0     0     0     0     0.05]

function myimp(A, pre2, pre3, pre4)
    pre2 = inv(A)
    mul!(pre3, pre2, S)
    mul!(pre4, pre3, pre2')
end

@benchmark myimp($A_real, $pre2, $pre3, $pre4)

loss = Loss([SemML(semobserved, [0.0], similar(start_val))])

diff_ana = SemAnalyticDiff(LBFGS(), Optim.Options(;show_trace = true),
            A, S, F, x, start_val)

imply = ImplySymbolic(A, S, F, θ, start_val)

imply(start_val)

imply_alloc = sem.ImplySymbolicAlloc(A, S, F, x, start_val)
imply_forward = sem.ImplySymbolicForward(A, S, F, x, start_val)

model_fin = Sem(semobserved, imply, loss, diff_fin)
model_rev = Sem(semobserved, imply_alloc, loss, diff_rev)
model_for = Sem(semobserved, imply_alloc, loss, diff_for)
model_for2 = Sem(semobserved, imply_forward, loss, diff_for)
model_ana = Sem(semobserved, imply, loss, diff_ana)

#Zygote.@nograd isposdef
#Zygote.@nograd Symmetric

@btime diff_fin = SemFiniteDiff("a", "b")

diff_fin = SemFiniteDiff(BFGS(), Optim.Options())

function loopsem(k, sem, semobserved, imply, loss, diff_fin)
    A = Vector{Any}(undef, k)
    for i in 1:k
        A[i] = deepcopy(Sem(semobserved, imply, loss, diff_fin))
    end
    return A
end

@benchmark loopsem(1000, model_fin, semobserved, imply, loss, diff_fin)

testmat = copy(test[1].imply.imp_cov)

test[1].imply.imp_cov .= zeros(11,11)

test[2].imply.imp_cov

test[1].diff.algorithm = "c"

model_fin

ForwardDiff.gradient(model_for, start_val)
FiniteDiff.finite_difference_gradient(model_fin, start_val)
Zygote.gradient(model_for, start_val)

grad = copy(start_val)
model_ana(nothing, grad, start_val)
grad ≈ ForwardDiff.gradient(model_for, start_val)

model_for.loss.functions[1].grad

output = copy(start_val)

model_fin(start_val)

typeof(model_fin.imply)

solution_fin = sem_fit(model_fin)
solution_for = sem_fit(model_for)
solution_for2 = sem_fit(model_for2)
solution_rev = sem_fit(model_rev)
solution_ana = sem_fit(model_ana)