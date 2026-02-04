############################################################################################
### Types
############################################################################################

# dummy SEM optimizer
"""
Empty placeholder for models that don't need
an optimizer part.

# Constructor

    SemOptimizerEmpty()
"""
struct SemOptimizerEmpty <: SemOptimizer{:Empty} end

############################################################################################
### Constructor
############################################################################################

SemOptimizer(::Val{:Empty}) = SemOptimizerEmpty()

SemOptimizer_impltype(::Val{:Empty}) = SemOptimizerEmpty

############################################################################################
### Recommended methods
############################################################################################

update_observed(optimizer::SemOptimizerEmpty, observed::SemObserved; kwargs...) = optimizer

############################################################################################
### Pretty Printing
############################################################################################

function Base.show(io::IO, struct_inst::SemOptimizerEmpty)
    print_type_name(io, struct_inst)
end
