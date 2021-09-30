##### weighted least squares
 struct SemWLS{Vt <: Union{AbstractArray, UniformScaling{Bool}},
        St <: AbstractArray} <: LossFunction
    V::Vt
    s::St
end


### Constructor
function SemWLS(observed::T, objective, grad; V = nothing) where {T <: SemObs}
    ind = CartesianIndices(observed.obs_cov)
    ind = filter(x -> (x[1] >= x[2]), ind)
    s = observed.obs_cov[ind]
    # compute V here
    if isnothing(V)
        D = duplication_matrix(observed.n_man)
        S = inv(observed.obs_cov)
        S = kron(S,S)
        V = 0.5*(D'*S*D)
    end
    return SemWLS(V, s)
end

### Loss
function (semwls::SemWLS)(par, model)
    diff = semwls.s - model.imply.imp_cov
    F = dot(diff, semwls.V, diff)
    return F
end


############## SWLS
struct SemSWLS{Vt <: Union{AbstractArray, UniformScaling{Bool}},
        St <: AbstractArray} <: LossFunction
    V::Vt
    sᵀV::St
    #s::St
end


### Constructor
function SemSWLS(observed::T, objective, grad; V = nothing) where {T <: SemObs}
    ind = CartesianIndices(observed.obs_cov)
    ind = filter(x -> (x[1] >= x[2]), ind)
    s = observed.obs_cov[ind]
    
    if isnothing(V)
        D = duplication_matrix(observed.n_man)
        S = inv(observed.obs_cov)
        S = kron(S,S)
        V = 0.5*(D'*S*D)
    end

    sᵀV = transpose(s)*V
    return SemSWLS(V, sᵀV)
end

### Loss
function (semswls::SemSWLS)(par, model)
    # s = semswls.s
    # V = semswls.V
    # G = model.imply.G
    outer = semswls.sᵀV*model.imply.G
    b = cholesky(Symmetric(model.imply.G'*semswls.V*model.imply.G))
    inner = b\outer'
    F = -outer*inner
    # F = transpose(s)*V*s - transpose(s)*V*G*inv(transpose(G)*V*G)*(transpose(G)*V*s)
#=     θ = inv(transpose(G)*V*G)*(transpose(G)*V*s)
    σ = G*θ
    diff = s - σ
    F = dot(diff, V, diff) =#
    return F
end