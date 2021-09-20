using ModelingToolkit, LinearAlgebra

ModelingToolkit.@variables s[1:5], σ[1:5], V[1:5, 1:5]

F = permutedims(s-σ)*V*(s-σ)

F = simplify(F)

nobs = 2
nlat = 1

### compute K

Sig = ones(Bool, nobs, nobs)
Sig = LowerTriangular(Sig)

ind = findall(Sig)

Es = []

for i in ind
    E = zeros(Bool, nobs,nobs)
    E[i] = true
    E = vec(E)
    push!(Es, E)
end

L = hcat(Es...)

K = L*inv(transpose(L)*L)

K = convert(BitArray, K)

# compute L's

Lat = ones(Bool, nlat, nlat)
Lat = LowerTriangular(Lat)

ind = findall(Lat)

Es = []
for i in ind
    E = zeros(Bool, nlat, nlat)
    E[i] = true
    E = vec(E)
    push!(Es, E)
end

L_lat = hcat(Es...)

Obse = zeros(Bool, nobs,nobs)
Obse[diagind(Obse)] .= true

ind = findall(Obse)

Es = []
for i in ind
    E = zeros(Bool, nobs, nobs)
    E[i] = true
    E = vec(E)
    push!(Es, E)
end

L_obs = hcat(Es...)

# Define Model

ModelingToolkit.@variables λ[1:2, 1:2], ε[1:6], ϕ[1:3]

Λ =[1 0
    λ[1,1] 0
    λ[1,2] 0
    0 1
    0 λ[2,1]
    0 λ[2,2]]

Θ = [ε[1] 0 0 0 0 0
    0 ε[2] 0 0 0 0
    0 0 ε[3] 0 0 0
    0 0 0 ε[4] 0 0
    0 0 0 0 ε[5] 0
    0 0 0 0 0 ε[6]]

Φ = [ϕ[1] ϕ[3]
    ϕ[3] ϕ[2]]

transpose(K)*(kron(Λ,Λ)*L_lat*L_obs)

G = transpose(K)*(kron(Λ,Λ)*L_lat)

G = simplify.(G)

G


### in RAM notation

F = zeros(2, 3)
F[diagind(F)] .= 1.0

kron(F, F)

ModelingToolkit.@variables λ[1:2], s[1:3]

A = [0 0 λ[1]
    0 0 λ[2]
    0 0 0]

A = I + A

A = simplify.(A)

kronprod = simplify.(kron(F, F)*kron(A, A))

kronprod

simplify.(kron(F*A, F*A))

Obse = zeros(Bool, nobs+nlat, nobs+nlat)
Obse[diagind(Obse)] .= true

ind = findall(Obse)

Es = []
for i in ind
    E = zeros(Bool, nobs+nlat, nobs+nlat)
    E[i] = true
    E = vec(E)
    push!(Es, E)
end

L_obs = hcat(Es...)

G = transpose(K)*kronprod*L_obs

G = simplify.(G)

simplify.(G*s)



### in RAM notation

F = zeros(2, 3)
F[diagind(F)] .= 1.0

kron(F, F)

ModelingToolkit.@variables λ[1:2], s[1:3], A[1:3, 1:3]

A = [0 0 λ[1]
    0 0 λ[2]
    0 0 0]

A = I + A

A = simplify.(A)

kronprod = simplify.(kron(F, F)*kron(A, A))

kronprod

simplify.(kron(F*A, F*A))

Obse = zeros(Bool, nobs+nlat, nobs+nlat)
Obse[diagind(Obse)] .= true

ind = findall(Obse)

Es = []
for i in ind
    E = zeros(Bool, nobs+nlat, nobs+nlat)
    E[i] = true
    E = vec(E)
    push!(Es, E)
end

L_obs = hcat(Es...)

G = transpose(K)*kronprod*L_obs

G = simplify.(G)

simplify.(G*s)


### in RAM notation

F = zeros(2, 3)
F[diagind(F)] .= 1.0

kron(F, F)

ModelingToolkit.@variables λ[1:2], s[1:3]

A = [0 0 λ[1]
    0 0 λ[2]
    0 0 0]

A = I + A

A = simplify.(A)

kronprod = simplify.(kron(F, F)*kron(A, A))

kronprod

simplify.(kron(F*A, F*A))

Obse = zeros(Bool, nobs+nlat, nobs+nlat)
Obse[diagind(Obse)] .= true

ind = findall(Obse)

Es = []
for i in ind
    E = zeros(Bool, nobs+nlat, nobs+nlat)
    E[i] = true
    E = vec(E)
    push!(Es, E)
end

L_obs = hcat(Es...)

G = transpose(K)*kronprod*L_obs

G = simplify.(G)

simplify.(G*s)



### in RAM notation
using SparseArrays

nobs = 9
nlat = 6

F = zeros(9, 15)
F[diagind(F)] .= 1.0
F = sparse(F)
#kron(F, F)

ModelingToolkit.@variables λ[1:12], ϕ[1:6, 1:6], σ[1:9]

A =[0 0 0 0 0 0 0 0 0 1.0 0 0 1 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 1 0 0
    0 0 0 0 0 0 0 0 0 0 0 1 1 0 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 1 0
    0 0 0 0 0 0 0 0 0 0 1 0 0 1 0
    0 0 0 0 0 0 0 0 0 0 0 1 0 1 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 0 1
    0 0 0 0 0 0 0 0 0 0 1 0 0 0 1
    0 0 0 0 0 0 0 0 0 0 0 1 0 0 1
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
]

A =[0 0 0 0 0 0 0 0 0 1 0 0 1 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 0.91 0 0
    0 0 0 0 0 0 0 0 0 0 0 1 0.92 0 0
    0 0 0 0 0 0 0 0 0 0.93 0 0 0 1 0
    0 0 0 0 0 0 0 0 0 0 0.96 0 0 0.76 0
    0 0 0 0 0 0 0 0 0 0 0 0.72 0 0.72 0
    0 0 0 0 0 0 0 0 0 0.87 0 0 0 0 1
    0 0 0 0 0 0 0 0 0 0 0.76 0 0 0 0.842
    0 0 0 0 0 0 0 0 0 0 0 0.97 0 0 0.767
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
]

A =[0 0 0 0 0 0 0 0 0 1 0 0 1 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 λ[7] 0 0
    0 0 0 0 0 0 0 0 0 0 0 1 λ[8] 0 0
    0 0 0 0 0 0 0 0 0 λ[1] 0 0 0 1 0
    0 0 0 0 0 0 0 0 0 0 λ[2] 0 0 λ[9] 0
    0 0 0 0 0 0 0 0 0 0 0 λ[3] 0 λ[10] 0
    0 0 0 0 0 0 0 0 0 λ[4] 0 0 0 0 1
    0 0 0 0 0 0 0 0 0 0 λ[5] 0 0 0 λ[11]
    0 0 0 0 0 0 0 0 0 0 0 λ[6] 0 0 λ[12]
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
]

A = sparse(A)

S = zeros(Num, nobs+nlat, nobs+nlat)
S[diagind(S)[1:9]] .= σ
S[13:15, 13:15] .= ϕ[4:6, 4:6]
S[10:12, 10:12] .= ϕ[1:3, 1:3]

θ = sparse(LowerTriangular(S)).nzval
θ

rank(A)


A = I + A

#A = simplify.(A)

A = inv(I-A)

kronprod = simplify.(kron(F, F)*kron(A, A))

kronprod = kron(F, F)*kron(A, A)

simplify.(kron(F*A, F*A))


Sig = ones(Bool, nobs, nobs)
Sig = LowerTriangular(Sig)

ind = findall(Sig)

Es = []

for i in ind
    E = zeros(Bool, nobs,nobs)
    E[i] = true
    E = vec(E)
    push!(Es, E)
end

L = hcat(Es...)

K = L*inv(transpose(L)*L)

K = convert(BitArray, K)
K = sparse(K)

Obse = zeros(Bool, nobs+nlat, nobs+nlat)
Obse[diagind(Obse)] .= true
Obse[13:15, 13:15] .= true
Obse[10:12, 10:12] .= true
Obse = LowerTriangular(Obse)
ind = findall(Obse)

Obse

Es = []
for i in ind
    E = zeros(Bool, nobs+nlat, nobs+nlat)
    E[i] = true
    E = vec(E)
    push!(Es, E)
end

L_obs = hcat(Es...)
L_obs = sparse(L_obs)

G = transpose(K)*kronprod*L_obs

G = simplify.(Matrix(G))

G = sparse(G)

reshape(simplify.(L_obs*θ), (15, 15))[10:15, 10:15]

simplify.(G*θ)

V = rand(45,45)
V = V*V'

test = transpose(G)*V*G

using BenchmarkTools

rank(test)

test2 = V[1:27, 1:27]

@benchmark pinv(test)

@benchmark inv(test2)

@benchmark pinv(test2)

pinv(test2) == inv(test2)