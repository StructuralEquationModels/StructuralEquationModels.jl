# Ordinary Maximum Likelihood Estimation

### Type
#= struct SemML{
        I <: AbstractArray,
        T <: AbstractArray,
        U, V,
        W <: Union{Nothing, AbstractArray}} <: LossFunction
    imp_inv::I
    mult::T # is this type known?
    objective::U
    grad::V
    meandiff::W
end =#

struct SemML{
        INV <: AbstractArray,
        C <: Union{AbstractArray, Nothing},
        L <: Union{AbstractArray, Nothing},
        M <: Union{AbstractArray, Nothing},
        M2 <: Union{AbstractArray, Nothing},
        U,
        V} <: LossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    mult::M
    logdets::L #logdets of implied covmats
    meandiff::M2
    #data_rowwise::DP  -> add to semobs
    objective::U
    grad::V
end

### Constructor
function SemML(observed::T, objective, grad) where {T <: SemObs}
    isnothing(observed.obs_mean) ?
        meandiff = nothing :
        meandiff = copy(observed.obs_mean)
    return SemML(
        copy(observed.obs_cov),
        nothing,
        copy(observed.obs_cov),
        nothing,
        meandiff,
        copy(objective),
        copy(grad),
        )
end

### Loss
# generic (not optimized)
function (semml::SemML)(par, model)
    B = copy(model.imply.imp_cov)
    C = similar(model.imply.imp_cov)

    a = cholesky!(Hermitian(B); check = false)
    if !isposdef(a)
        F = Inf
    else
        mul!(C, inv(a), model.observed.obs_cov)
        F = logdet(a) +
             tr(C)
    end
    #B = nothing
    #C = nothing
    #a = nothing
    return F
end

# maximum speed for finite differences
function (semml::SemML)(par, model::Sem{O, I, L, D}) where
            {O <: SemObs, L <: Loss, I <: Imply, D <: SemFiniteDiff}
    semml.inverses .= model.imply.imp_cov
    a = cholesky!(Hermitian(semml.imp_inv); check = false)
    if !isposdef(a)
        F = Inf
    else
        ld = logdet(a)
        semml.inverses .= LinearAlgebra.inv!(a)
        #inv_cov = inv(a)
        mul!(semml.mult, semml.inverses, model.observed.obs_cov)
        if !isnothing(model.imply.imp_mean)
            @. semml.meandiff = model.observed.obs_mean - model.imply.imp_mean
            F_mean = semml.meandiff'*semml.imp_inv*semml.meandiff
        else
            F_mean = zero(eltype(par))
        end
        F = ld +
            tr(semml.mult) + F_mean
    end
    return F
end

# analytic differentiation
function (semml::SemML)(par, model::Sem{O, I, L, D}, E, G) where
            {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}

    a = cholesky(Hermitian(model.imply.imp_cov); check = false)
    if !isposdef(a)
        if E != nothing
            semml.objective .= Inf
        end
        if G != nothing
            semml.grad .= 0.0
        end
    else
        ld = logdet(a)
        inv_cov = inv(a)
        if E != nothing
            mul!(semml.mult, inv_cov, model.observed.obs_cov)
            semml.objective .= ld + tr(semml.mult)
        end
        if G != nothing
            model.diff.B!(model.diff.B, par) # B = inv(I-A)
            model.diff.E!(model.diff.E, par) # E = B*S*B'
            let B = model.diff.B, E = model.diff.E,
                Σ_inv = inv_cov, F = model.diff.F,
                D = model.observed.obs_cov
                for i = 1:size(par, 1)
                    S_der = sparse(model.diff.S_ind_vec[i]..., model.diff.matsize...)
                    A_der = sparse(model.diff.A_ind_vec[i]..., model.diff.matsize...)

                    term = F*B*A_der*E*F'
                    Σ_der = Array(F*B*S_der*B'F' + term + term')

                    semml.grad[i] = tr(Σ_inv*Σ_der) + tr((-Σ_inv)*Σ_der*Σ_inv*D)
                end
            end
        end
    end
end
