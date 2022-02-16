Base.@kwdef struct RAMMatrices
    A
    S
    F
    M = nothing
    parameters
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, ram_matrices::RAMMatrices)
    print_type_name(io, ram_matrices)
    print_field_types(io, ram_matrices)
end