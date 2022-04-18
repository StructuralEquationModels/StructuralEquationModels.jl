##### weighted least squares

############################################################################
### Types
############################################################################

struct SemWLS{Vt, St, B, C, FT, GT, HT} <: SemLossFunction
    V::Vt
    s::St
    approx_H::B
    V_μ::C

    objective::FT
    gradient::GT
    hessian::HT
end

############################################################################
### Constructors
############################################################################

function SemWLS(;observed, n_par, V = nothing, meanstructure = false, V_μ = nothing, approx_H = false, parameter_type = Float64, kwargs...)
    ind = CartesianIndices(obs_cov(observed))
    ind = filter(x -> (x[1] >= x[2]), ind)
    s = obs_cov(observed)[ind]

    # compute V here
    if isnothing(V)
        D = duplication_matrix(n_man(observed))
        S = inv(obs_cov(observed))
        S = kron(S, S)
        V = 0.5*(D'*S*D)
    end

    if meanstructure
        if isnothing(V_μ)
            V_μ = inv(obs_cov(observed))
        end
    else
        V_μ = nothing
    end

    return SemWLS(
        V, 
        s, 
        approx_H, 
        V_μ,

        zeros(parameter_type, 1),
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par))
end

############################################################################
### functors
############################################################################

function (semwls::SemWLS)(par, F, G, H, model)
    σ_diff = semwls.s - Σ(imply(model))
    if isnothing(semwls.V_μ)
    # without meanstructure
        if G && H
            J = (-2*(σ_diff)'*semwls.V)'
            gradient = ∇Σ(imply(model))'*J
            semwls.gradient .= gradient
            hessian = 2*∇Σ(imply(model))'*semwls.V*∇Σ(imply(model))
            if !semwls.approx_H
                ∇²Σ_function(imply(model))(∇²Σ(imply(model)), J, par)
                hessian += ∇²Σ(imply(model)) 
            end
            semwls.hessian .= hessian
        end
        if !G && H
            hessian = 2*∇Σ(imply(model))'*semwls.V*∇Σ(imply(model))
            if !semwls.approx_H
                J = (-2*(σ_diff)'*semwls.V)'
                ∇²Σ_function(imply(model))(∇²Σ(imply(model)), J, par)
                hessian += ∇²Σ(imply(model))
            end
            semwls.hessian .= hessian
        end
        if G && !H
            gradient = (-2*(σ_diff)'*semwls.V*∇Σ(imply(model)))'
            semwls.gradient .= gradient
        end
        if F
            semwls.objective[1] = dot(σ_diff, semwls.V, σ_diff)   
        end
    else
    # with meanstructure
    μ_diff = obs_mean(observed(model)) - μ(imply(model))
        if H throw(DomainError(H, "hessian of WLS with meanstructure is not available")) end
        if G
            gradient = -2*(σ_diff'*semwls.V*∇Σ(imply(model)) + μ_diff'*semwls.V_μ*∇μ(imply(model)))'
            semwls.gradient .= gradient
        end
        if F
            semwls.objective[1] = σ_diff'*semwls.V*σ_diff + μ_diff'*semwls.V_μ*μ_diff
        end
    end
end

############################################################################
### Recommended methods
############################################################################

objective(lossfun::SemWLS) = lossfun.objective
gradient(lossfun::SemWLS) = lossfun.gradient
hessian(lossfun::SemWLS) = lossfun.hessian

update_observed(lossfun::SemWLS, observed::SemObs; kwargs...) = SemWLS(;observed = observed, kwargs...)

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemWLS)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end