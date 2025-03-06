"""
    params!(out::AbstractVector, partable::ParameterTable,
                  col::Symbol = :estimate)

Extract parameter values from the `col` column of `partable`
into the `out` vector.

The `out` vector should be of `nparams(partable)` length.
The *i*-th element of the `out` vector will contain the
value of the *i*-th parameter from `params_labels(partable)`.

Note that the function combines the duplicate occurences of the
same parameter in `partable` and will raise an error if the
values do not match.
"""
function params!(
    out::AbstractVector,
    partable::ParameterTable,
    col::Symbol = :estimate,
)
    (length(out) == nparams(partable)) || throw(
        DimensionMismatch(
            "The length of parameter values vector ($(length(out))) does not match the number of parameters ($(nparams(partable)))",
        ),
    )
    param_index = param_indices(partable)
    param_values_col = partable.columns[col]
    for (i, param) in enumerate(partable.columns[:param])
        (param == :const) && continue
        param_ind = get(param_index, param, nothing)
        @assert !isnothing(param_ind) "Parameter table contains unregistered parameter :$param at row #$i"
        val = param_values_col[i]
        if !isnan(out[param_ind])
            @assert isequal(out[param_ind], val) "Parameter :$param value at row #$i ($val) differs from the earlier encountered value ($(out[param_ind]))"
        else
            out[param_ind] = val
        end
    end
    return out
end

"""
    params(out::AbstractVector, col::Symbol = :estimate)

Extract parameter values from the `col` column of `partable`.

Returns the values vector. The *i*-th element corresponds to
the value of *i*-th parameter from `params_label(partable)`.

Note that the function combines the duplicate occurences of the
same parameter in `partable` and will raise an error if the
values do not match.
"""
params(partable::ParameterTable, col::Symbol = :estimate) =
    params!(fill(NaN, nparams(partable)), partable, col)
