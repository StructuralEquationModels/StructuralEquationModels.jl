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
    params_col = partable.columns[col]
    for (i, label) in enumerate(partable.columns[:label])
        (label == :const) && continue
        param_ind = get(param_index, label, nothing)
        @assert !isnothing(param_ind) "Parameter table contains unregistered parameter :$param at row #$i"
        param = params_col[i]
        if !isnan(out[param_ind])
            @assert isequal(out[param_ind], param) "Parameter :$label value at row #$i ($param) differs from the earlier encountered value ($(out[param_ind]))"
        else
            out[param_ind] = param
        end
    end
    return out
end

"""
    params(x::ParameterTable, col::Symbol = :estimate)

Extract parameter values from the `col` column of `partable`.

Returns the values vector. The *i*-th element corresponds to
the value of *i*-th parameter from `params_label(partable)`.

Note that the function combines the duplicate occurences of the
same parameter in `partable` and will raise an error if the
values do not match.
"""
params(partable::ParameterTable, col::Symbol = :estimate) =
    params!(fill(NaN, nparams(partable)), partable, col)

"""
    coef(x::ParameterTable)

For a `SEM`, this function is equivalent to `params(x)`.
It returns the parameters for the given model.
"""
coef(x::ParameterTable) = params(x)

"""
    coefnames(x::ParameterTable)
To maintain compatibility with the `lavaan` package, this function is a synonym for `param_labels(x)`.
"""
coefnames(x::ParameterTable) = param_labels(x)

"""
    nobs(model::AbstractSem) -> Int

Synonymous to [*nsamples*](@ref nsamples).
"""
nobs(model::AbstractSem) = nsamples(model)

coeftable(model::AbstractSem; level::Real=0.95) = throw(MethodError(x, "StructuralEquationModels does not support the `CoefTable` interface; see [`ParameterTable`](@ref) instead."))