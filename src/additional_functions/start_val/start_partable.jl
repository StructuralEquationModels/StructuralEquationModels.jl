# splice model and loss functions
function start_parameter_table(model::Union{Sem, SemForwardDiff, SemFiniteDiff}; kwargs...)
    return start_parameter_table(
        model.observed, 
        model.imply,
        model.diff, 
        model.loss.functions...;
        kwargs...)
end

# RAM(Symbolic)
function start_parameter_table(observed, imply::Union{RAM, RAMSymbolic}, diff, args...; kwargs...)
    return start_parameter_table(
        imply.ram_matrices;
        kwargs...)
end

function start_parameter_table(ram_matrices::RAMMatrices; parameter_table::ParameterTable, kwargs...)
    
    start_val = zeros(0)
    
    for identifier_ram in ram_matrices.parameters
        found = false
        for (i, identifier_table) in enumerate(parameter_table.identifier)
            if identifier_ram == identifier_table
                push!(start_val, parameter_table.start[i])
                found = true
                break
            end
        end
        if !found throw(ErrorException("At least one parameter could not be found in the parameter table.")) end
    end

    return start_val

end