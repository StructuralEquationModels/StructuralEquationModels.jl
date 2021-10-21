using Symbolics, SparseArrays, LinearAlgebra

nfact = 10
nitem = 5

nobs = nitem*nfact
ntot = nobs+nfact

npar = Int64(nobs + (nfact+1)*nfact/2 + nobs - nfact)

@Symbolics.variables x[1:npar]

x = [x[i] for i in 1:size(x, 1)]
#cov = [cov[i,j] for i in 1:size(cov, 1) for j in 1:size(cov, 2)]
S = zeros(Num, ntot, ntot)
S[diagind(S)[1:nobs]] .= x[1:nobs]
xind = nobs
for i in 1:nfact
    for j in i:nfact
        global xind = xind+1
        ind = nobs+i
        jnd = nobs+j
        S[ind, jnd] = x[xind]
        S[jnd, ind] = x[xind]
    end
end

F = zeros(Num, nobs, ntot)

F[diagind(F)] .= 1

A = zeros(Num, ntot, ntot)


for i in 1:nfact
    for j in 2:nitem
        global xind = xind+1
        jnd = i+nobs
        ind = j+(i-1)*nitem
        A[ind, jnd] = x[xind]
    end
end

for i in 1:nfact
    A[1+(i-1)*nitem, nobs+i] = 1
end


S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)

par = [x[i] for i = 1:npar]
invia = I + A + A^2 + A^3 + A^4
imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
imp_cov_sym = Array(imp_cov_sym)
imp_cov_sym = Symbolics.simplify.(imp_cov_sym)

imp_fun = Symbolics.build_function(imp_cov_sym, par, target=Symbolics.CTarget())
print(imp_fun)
