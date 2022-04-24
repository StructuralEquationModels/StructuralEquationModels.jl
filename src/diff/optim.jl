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

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemDiffOptim)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end