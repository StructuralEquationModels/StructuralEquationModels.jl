############################################################################
### Types
############################################################################

struct ImplyEmpty{V} <: SemImply
    start_val::V
end

############################################################################
### Constructors
############################################################################

function ImplyEmpty(;
        ram_matrices = nothing,
        start_val = start_fabin3,
        kwargs...)

        if !isa(start_val, Vector)
            start_val = start_val(;ram_matrices = ram_matrices, kwargs...)
        end

        return ImplyEmpty(start_val)
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