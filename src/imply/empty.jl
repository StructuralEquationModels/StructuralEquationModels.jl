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
                identifier = Dict{Symbol, Int64}(ram_matrices.identifier .=> 1:length(ram_matrices.identifier))
            elseif specification isa ParameterTable
                ram_matrices = RAMMatrices!(specification)
                identifier = Dict{Symbol, Int64}(ram_matrices.identifier .=> 1:length(ram_matrices.identifier))
            else
                @error "The RAM constructor does not know how to handle your specification object. 
                \n Please specify your model as either a ParameterTable or RAMMatrices."
            end
            start_val = start_val(;ram_matrices = ram_matrices, specification = specification, kwargs...)
        end

        return ImplyEmpty(start_val, identifier)
end

############################################################################
### functors
############################################################################

function (imply::ImplyEmpty)(par, F, G, H, model) end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::ImplyEmpty)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end

############################################################################
### Identifier
############################################################################

identifier(imply::ImplyEmpty) = imply.identifier