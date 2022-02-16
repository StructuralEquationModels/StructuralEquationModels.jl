struct SemDiffNLopt{A, B} <: SemDiff
    algorithm::A
    options::B
end

SemDiffNLopt(;algorithm = :LD_LBFGS, options = nothing, kwargs...) = SemDiffNLopt(algorithm, options)

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::SemDiffNLopt)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end