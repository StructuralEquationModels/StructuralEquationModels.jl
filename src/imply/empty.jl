############################################################################
### Types
############################################################################

struct ImplyEmpty{V, V2} <: SemImply
    identifier::V2
    n_par::V
end

############################################################################
### Constructors
############################################################################

function ImplyEmpty(;
        specification,
        kwargs...)

        ram_matrices = RAMMatrices(specification)
        identifier = StructuralEquationModels.identifier(ram_matrices)
        
        n_par = length(ram_matrices.parameters)

        return ImplyEmpty(identifier, n_par)
end

############################################################################
### methods
############################################################################

objective!(imply::ImplyEmpty, par, model) = nothing
gradient!(imply::ImplyEmpty, par, model) = nothing
hessian!(imply::ImplyEmpty, par, model) = nothing

############################################################################
### Recommended methods
############################################################################

identifier(imply::ImplyEmpty) = imply.identifier
n_par(imply::ImplyEmpty) = imply.n_par

update_observed(imply::ImplyEmpty, observed::SemObs; kwargs...) = imply

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::ImplyEmpty)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end