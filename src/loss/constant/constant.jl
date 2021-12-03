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

function SemConstant(C, n_par; parameter_type = Float64)
    return SemConstant(
        [C],
        zeros(parameter_type, n_par),
        zeros(parameter_type, n_par, n_par))
end

############################################################################
### functors
############################################################################

function (constant::SemConstant)(par, F, G, H, model) end