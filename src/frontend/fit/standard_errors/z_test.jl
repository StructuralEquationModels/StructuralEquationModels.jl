_doc_z_test = """
    (1) z_test(fitted, se)

    (2) z_test!(partable, fitted, se, name = :p_value)

Return two-sided p-values from a z-test for each model parameter.

Tests the null hypothesis that each parameter is zero using the test statistic
`z = estimate / se`, which is compared against a standard normal distribution.
`z_test!` additionally writes the result into `partable`.

# Arguments
- `fitted`: a fitted SEM.
- `se`: standard errors for each parameter, e.g. from [`se_hessian`](@ref) or
  [`se_bootstrap`](@ref).
- `partable`: a [`ParameterTable`](@ref) to write p-values to.
- `name`: column name for the p-values in `partable`. Defaults to `:p_value`.

# Returns
- a vector of p-values.
"""

@doc "$(_doc_z_test)"
function z_test(fitted, se)
    dev = solution(fitted) ./ se
    dist = Normal(0, 1)
    p = 2*ccdf.(dist, abs.(dev))
    return p
end

@doc "$(_doc_z_test)"
function z_test!(partable, fitted, se, name = :p_value)
    p = z_test(fitted, se)
    update_partable!(partable, name, fitted, p)
    return p
end
