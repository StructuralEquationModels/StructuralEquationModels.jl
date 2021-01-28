# minimal working example for fitting a sem

using LinearAlgebra, Optim, Zygote, ReverseDiff, AutoGrad, Yota, 
    FiniteDiff, BenchmarkTools


pars = [1.0, 2]
A = rand(2,2); A = A*A'
pre = copy(A)
prod = copy(A)
a = cholesky(A)

function floss(pars, A, a, pre, prod)
    A[1] = pars[2]
    A[2] = pars[1]
    A[3] = pars[1]
    A[4] = pars[2]
    a = cholesky!(A; check = false)
    if !isposdef(a) return Inf end 
    pre = LinearAlgebra.inv(a)
    #mul!(prod, pre, A)
    F = logdet(a) + tr(pre)
end

function floss2(pars, A, a, pre, prod)
    A = [pars[2] pars[1]
        pars[1] pars[2]]
    B = copy(A)    
    a = cholesky!(A)
    pre = LinearAlgebra.inv(a)
    prod = pre*B
    F = logdet(a) + tr(prod)
end

floss(pars, A, a, pre, prod)
floss2(pars, A, a, pre, prod)

@benchmark floss(pars, A, a, pre, prod)



# reverse diff
# Zygote; ReverseDiff; AutoGrad; Yota


# FiniteDiff
fgrad = 
    FiniteDiff.finite_difference_jacobian(
        x -> floss2(x, A, a, pre, prod), pars)

# Zygote        
zgrad = 
    Zygote.gradient(x -> floss2(x, A, a, pre, prod), pars)

x = Param([1,2,3])		# user declares parameters
x => P([1,2,3])			# they are wrapped in a struct
value(x) => [1,2,3]		# we can get the original value
sum(abs2,x) => 14		# they act like regular values outside of differentiation
y = @diff sum(abs2,x)	        # if you want the gradients
y => T(14)			# you get another struct
value(y) => 14			# which represents the same value
grad(y,x) => [2,4,6]

a_pars = Param(pars)
agrad = @diff floss2(a_pars, A, a, pre, prod)

#Yota
ygrad = 
    Yota.grad(x -> floss2(x, A, a, pre, prod), pars)


# ReverseDiff    
# pre-record a GradientTape for `f` using inputs of shape 100x100 with Float64 elements
const floss_tape = 
    ReverseDiff.GradientTape(x -> floss(x, A, a, pre, prod), (pars))

const floss_tape22 = 
    ReverseDiff.GradientTape(x -> floss2(x, A, a, pre, prod), (pars))    
# compile `f_tape` into a more optimized representation

const compiled_f_tape = ReverseDiff.compile(floss_tape)
const compiled_f_tape22 = ReverseDiff.compile(floss_tape22)

results = similar(pars)
ReverseDiff.gradient!(results, compiled_f_tape, pars)
ReverseDiff.gradient!(results, compiled_f_tape22, pars)

### => Yota and AutoGrad give errors

#########################
function floss_r(pars, A, a, pre, prod)
    A = [pars[2] pars[1]
        pars[1] pars[2]]
    B = copy(A)    
    a = cholesky!(A)
    pre = LinearAlgebra.inv(a)
    mul!(prod, pre, B)
    F = logdet(a) + tr(prod)
end

function floss_z(pars, A, a, pre, prod)
    A = [pars[2] pars[1]
        pars[1] pars[2]]
    B = copy(A)    
    a = cholesky(A)
    pre = LinearAlgebra.inv(a)
    prod = pre*B
    F = logdet(a) + tr(prod)
end



const floss_tape22222 = 
    ReverseDiff.GradientTape(x -> floss_r(x, A, a, pre, prod), (pars))    
# compile `f_tape` into a more optimized representation
const compiled_f_tape22222 = ReverseDiff.compile(floss_tape22222)

results = similar(pars)

ReverseDiff.gradient!(results, compiled_f_tape22222, pars)

zgrad = 
    Zygote.gradient(x -> floss_z(x, A, a, pre, prod), pars)


#########################
function floss_r(pars, A, a, pre, prod)
    A = [pars[2] pars[1]
        pars[1] pars[2]]
    #B = copy(A)    
    a = cholesky!(A; check = false)
    if !isposdef(a) return Inf end 
    pre = LinearAlgebra.inv(a)
    #mul!(prod, pre, B)
    F = logdet(a) + tr(pre)
end

@btime optimize(x -> floss_r(x, A, a, pre, prod), pars, BFGS())

fgrad = 
    FiniteDiff.finite_difference_jacobian(
        x -> floss_r(x, A, a, pre, prod), pars)

const floss_t = 
    ReverseDiff.GradientTape(x -> floss_r(x, A, a, pre, prod), (pars))    
# compile `f_tape` into a more optimized representation
const compiled_f_t= ReverseDiff.compile(floss_t)

results = similar(pars)

g(G, x) = ReverseDiff.gradient!(G, compiled_f_t, x)

@btime optimize(x -> floss(x, A, a, pre, prod), g, pars, BFGS())

@benchmark g(results, pars)
@benchmark floss_r(pars, A, a, pre, prod)



function floss_z(pars, A, a, pre, prod)
    A = [pars[2] pars[1]
        pars[1] pars[2]]
    #B = copy(A)    
    if !isposdef(A) return Inf end
    pre = LinearAlgebra.inv(A)
    #prod = pre*B
    F = logdet(A) + tr(pre)
end

floss_z(pars, A, a, pre, prod)

zgrad = 
    Zygote.gradient(x -> floss_z(x, A, a, pre, prod), pars)

function g_z(G, x) 
    check = Zygote.gradient(x -> floss_z(x, A, a, pre, prod), x)[1]
    isnothing(check) ? G .= 0.0 : G.= check[1]
end

@btime optimize(x -> floss(x, A, a, pre, prod), g_z, pars, BFGS())    

### with build function

using ModelingToolkit

@variables v[1:2]
A = [v[2] v[1]
    v[1] v[2]]

b, b_ = eval.(ModelingToolkit.build_function(A, v))    


b(pars)
B = zeros(2,2)
b_(B, pars)

function floss_r(pars, A, a, pre, prod)
    A = b(pars)
    #B = copy(A)    
    a = cholesky!(Hermitian(A); check = false)
    if !isposdef(a) return Inf end
    ld = logdet(a)
    pre = LinearAlgebra.inv(a)
    #mul!(prod, pre, B)
    F = ld + tr(pre)
end

floss_r(pars, B, a, pre, prod)

const floss_t_5 = 
    ReverseDiff.GradientTape(x -> floss_r(x, B, a, pre, prod), (pars))    
# compile `f_tape` into a more optimized representation
const compiled_f_t_4 = ReverseDiff.compile(floss_t_4)

results = similar(pars)
ReverseDiff.gradient!(results, compiled_f_t_4, pars)

#### Test generated function vs matrix multiplication

#matrix

function fmat(x)
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

    invia = inv(I-A)    
    implied = F*invia*S*invia'*F'
    return logdet(implied)
end

# generated 
using ModelingToolkit, SparseArrays, sem

@ModelingToolkit.variables x[1:31]

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

invia = sem.neumann_series(A)

impcov = F*invia*S*permutedims(invia)*permutedims(F)

impcov = Array(impcov)

impcov = simplify.(impcov)

f_ = eval(ModelingToolkit.build_function(impcov, x)[1])

function fgen(par)
    impcov = f_(par)
    return tr(impcov)
end

### test

using BenchmarkTools

start_val = vcat(
    fill(1.0, 11),
    fill(0.05, 3),
    fill(0.0, 6),
    fill(1.0, 8),
    fill(0, 3)
    )

@btime fmat(start_val)

@btime fgen(start_val)


const mat_t = 
    ReverseDiff.GradientTape(x -> fmat(x), (start_val))    
# compile `f_tape` into a more optimized representation
const compiled_mat_t= ReverseDiff.compile(mat_t)

results = similar(start_val)

gmat(G, x) = ReverseDiff.gradient!(G, compiled_mat_t, x)


const gen_t3 = 
    ReverseDiff.GradientTape(x -> fgen(x), (start_val))    
# compile `f_tape` into a more optimized representation
const compiled_gen_t2 = ReverseDiff.compile(gen_t3)

results = similar(start_val)

ggen(G, x) = ReverseDiff.gradient!(G, compiled_gen_t2, x)


@btime ggen(results, start_val)

@btime gmat(results, start_val)


using ForwardDiff

@btime ForwardDiff.gradient(fgen, start_val)

FiniteDiff.finite_difference_gradient(fgen, start_val)

B = rand(100,100)

@btime inv(B)

#193.399 μs

Threads.nthreads()



@sync for i in 1:length(imply)
    @spawn for j in 1:length(imply[i] -> length(observed))
        F[k] = model[k](parameter)
    end
    @spawn for j in 1:length(imply[i] -> length(observed))
    end
    @spawn for j in 1:length(imply[i] -> length(observed))
    end
    @spawn for j in 1:length(imply[i] -> length(observed))
    end
end

rand(10,10)