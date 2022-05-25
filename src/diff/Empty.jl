############################################################################
### Types
############################################################################
"""
Subtype of `SemDIff` that implements an empty placeholder for models that don't need
a diff part.

# Constructor

    SemDiffEmpty()
"""
struct SemDiffEmpty <: SemDiff end

############################################################################
### Constructor
############################################################################

# SemDiffEmpty(;kwargs...) = SemDiffEmpty()

############################################################################
### Recommended methods
############################################################################

update_observed(diff::SemDiffEmpty, observed::SemObs; kwargs...) = diff

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemDiffEmpty)
    StructuralEquationModels.print_type_name(io, struct_inst)
end

