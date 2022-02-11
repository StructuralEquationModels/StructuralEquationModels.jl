# constant loss function for comparability with other packages

############################################################################
### Types
############################################################################

struct SemConstant{FT, GT, HT} <: SemLossFunction
    F::FT
    G::GT
    H::HT
end

############################################################################
### Constructors
############################################################################

function SemConstant(;constant_loss, n_par, parameter_type = Float64, kwargs...)
    return SemConstant(
        [constant_loss],
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par))
end

############################################################################
### functors
############################################################################

function (constant::SemConstant)(par, F, G, H, model) end