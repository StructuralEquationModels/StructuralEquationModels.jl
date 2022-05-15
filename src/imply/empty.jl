############################################################################
### Types
############################################################################

struct ImplyEmpty{V, V2} <: SemImply
    start_val::V
    identifier::V2
end

############################################################################
### Constructors
############################################################################

function ImplyEmpty(;
        specification = nothing,
        start_val = start_fabin3,
        kwargs...)

        if !isa(start_val, Vector)
            if specification isa RAMMatrices
                ram_matrices = specification
                identifier = StructuralEquationModels.identifier(ram_matrices)
            elseif specification isa ParameterTable
                ram_matrices = RAMMatrices!(specification)
                identifier = StructuralEquationModels.identifier(ram_matrices)
            else
                throw(ErrorException("The RAM constructor does not know how to handle your specification object. 
                \n Please specify your model as either a ParameterTable or RAMMatrices."))
            end
            start_val = start_val(;ram_matrices = ram_matrices, specification = specification, kwargs...)
        end

        return ImplyEmpty(start_val, identifier)
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