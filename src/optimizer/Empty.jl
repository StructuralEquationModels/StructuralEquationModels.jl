############################################################################################
### Types
############################################################################################

# dummy SEM optimizer
struct SemOptimizerEmpty <: SemOptimizer{:Empty} end

############################################################################################
### Constructor
############################################################################################

"""
    SemOptimizer(engine = :Empty)

Constructs a dummy optimizer for models that don't need it.
"""
SemOptimizer(::Val{:Empty}) = SemOptimizerEmpty()

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
