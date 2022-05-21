############################################################################
### Types and Constructor
############################################################################

mutable struct SemDiffOptim{A, B} <: SemDiff
    algorithm::A
    options::B
end

SemDiffOptim(;algorithm = LBFGS(), options = Optim.Options(;f_tol = 1e-10, x_tol = 1.5e-8), kwargs...) = SemDiffOptim(algorithm, options)

############################################################################
### Recommended methods
############################################################################

update_observed(diff::SemDiffOptim, observed::SemObs; kwargs...) = diff

############################################################################
### additional methods
############################################################################

algorithm(diff::SemDiffOptim) = diff.algorithm
options(diff::SemDiffOptim) = diff.options