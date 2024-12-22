############################################################################################
### Types
############################################################################################
"""
Empty placeholder for models that don't need
an optimizer part.

# Constructor

    SemOptimizerEmpty()

# Extended help

## Implementation

Subtype of `SemOptimizer`.
"""
struct SemOptimizerEmpty <: SemOptimizer{:Empty} end

############################################################################################
### Constructor
############################################################################################

SemOptimizer{:Empty}() = SemOptimizerEmpty()

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
