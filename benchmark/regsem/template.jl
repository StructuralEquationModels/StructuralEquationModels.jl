# Template to define new Loss Functions

############################################################################
### Type
############################################################################

struct SemRidge{FT, GT, HT} <: SemLossFunction
    α

    F::FT
    G::GT
    H::HT
end

############################################################################
### Constructor
############################################################################

function SemRidge(α, n_par; parameter_type = Float64)

    return SemRidge(
        α,

        zeros(parameter_type, 1),
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par))
end

############################################################################
### functors
############################################################################

function (ridge::SemRidge)(parameters, F, G, H, model)

    if F
        F = ridge.α*norm(parameters)^2
        mytype.F[1] = F
    end

    if G
        G = 2*ridge.α*parameters
        mytype.G .= G
    end

    if H
        stop("hessian for MyType is not implemented")
    end
    
end