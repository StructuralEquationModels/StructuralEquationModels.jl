# check if params vector correctly matches the parameter references (from the ParameterTable)
function check_param_labels(
    param_labels::AbstractVector{Symbol},
    param_refs::Union{AbstractVector{Symbol}, Nothing},
)
    dup_params = nonunique(param_labels)
    isempty(dup_param_labels) ||
        throw(ArgumentError("Duplicate parameter labels detected: $(join(dup_param_labels, ", "))"))
    any(==(:const), param_labels) &&
        throw(ArgumentError("Parameters constain reserved :const name"))

    if !isnothing(param_refs)
        # check if all references parameters are present
        all_refs = Set(id for id in param_refs if id != :const)
        undecl_params = setdiff(all_refs, param_labels)
        if !isempty(undecl_params)
            throw(
                ArgumentError(
                    "The following $(length(undecl_params)) parameters present in the table, but are not declared: " *
                    join(sort!(collect(undecl_params))),
                ),
            )
        end
    end

    return nothing
end

function check_vars(vars::AbstractVector{Symbol}, nvars::Union{Integer, Nothing})
    isnothing(nvars) ||
        length(vars) == nvars ||
        throw(
            DimensionMismatch(
                "variables length ($(length(vars))) does not match the number of columns in A matrix ($nvars)",
            ),
        )
    dup_vars = nonunique(vars)
    isempty(dup_vars) ||
        throw(ArgumentError("Duplicate variables detected: $(join(dup_vars, ", "))"))

    return nothing
end
