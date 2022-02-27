struct SemDiffProximal{A, B, C} <: SemDiff
    algorithm::A
    options::B
    g::C
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemDiffOptim)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end