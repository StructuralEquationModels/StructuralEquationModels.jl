############################################################################
### Types
############################################################################

struct SemDiffNLopt{A, A2, B, B2, C} <: SemDiff
    algorithm::A
    local_algorithm::A2
    options::B
    local_options::B2
    equality_constraints::C
    inequality_constraints::C
end

Base.@kwdef mutable struct NLoptConstraint
    f
    tol = 0.0
end

############################################################################
### Constructor
############################################################################

function SemDiffNLopt(;
        algorithm = :LD_LBFGS,
        local_algorithm = nothing, 
        options = Dict{Symbol, Any}(),
        local_options = Dict{Symbol, Any}(), 
        equality_constraints = Vector{NLoptConstraint}(), 
        inequality_constraints = Vector{NLoptConstraint}(), 
        kwargs...)
    applicable(iterate, equality_constraints) || (equality_constraints = [equality_constraints])
    applicable(iterate, inequality_constraints) || (inequality_constraints = [inequality_constraints])
    return SemDiffNLopt(algorithm, local_algorithm, options, local_options, equality_constraints, inequality_constraints)
end

############################################################################
### Recommended methods
############################################################################

update_observed(diff::SemDiffNLopt, observed::SemObs; kwargs...) = diff

############################################################################
### additional methods
############################################################################

algorithm(diff::SemDiffNLopt) = diff.algorithm
local_algorithm(diff::SemDiffNLopt) = diff.local_algorithm
options(diff::SemDiffNLopt) = diff.options
local_options(diff::SemDiffNLopt) = diff.local_options
equality_constraints(diff::SemDiffNLopt) = diff.equality_constraints
inequality_constraints(diff::SemDiffNLopt) = diff.inequality_constraints

