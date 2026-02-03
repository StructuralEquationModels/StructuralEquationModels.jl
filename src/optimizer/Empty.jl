############################################################################################
### Types
############################################################################################
"""
Empty placeholder for models that don't need
an optimizer part.

# Constructor

    SemOptimizerEmpty()
"""
struct SemOptimizerEmpty <: SemOptimizer{:Empty} end

sem_optimizer_subtype(::Val{:Empty}) = SemOptimizerEmpty

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
