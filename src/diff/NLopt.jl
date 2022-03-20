############################################################################
### Types
############################################################################

struct SemDiffNLopt{A, B} <: SemDiff
    algorithm::A
    options::B
    equality_constraints
    inequality_constraints
end

function SemDiffNLopt(;algorithm = :LD_LBFGS, options = nothing, equality_constraints = [], inequality_constraints = [], kwargs...) 
    applicable(iterate, equality_constraints) || equality_constraints = [equality_constraints]
    applicable(iterate, inequality_constraints) || equality_constraints = [equality_constraints]
    return SemDiffNLopt(algorithm, options, equality_constraints, inequality_constraints)
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemDiffNLopt)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end

############################################################################
### Options
############################################################################

Base.@kwdef mutable struct NLoptConstraint
    f
    tol = 0.0
end