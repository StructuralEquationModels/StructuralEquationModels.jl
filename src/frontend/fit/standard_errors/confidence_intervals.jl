_doc_normal_CI = """
    (1) normal_CI(fitted, se; α = 0.05, name_lower = :ci_lower, name_upper = :ci_upper)

    (2) normal_CI!(partable, fitted, se; α = 0.05, name_lower = :ci_lower, name_upper = :ci_upper)

Return normal-theory confidence intervals for all model parameters.
`normal_CI!` additionally writes the result into `partable`.

# Arguments
- `fitted`: a fitted SEM.
- `se`: standard errors for each parameter, e.g. from [`se_hessian`](@ref) or
  [`se_bootstrap`](@ref).
- `partable`: a [`ParameterTable`](@ref) to write confidence intervals to.
- `α`: significance level. Defaults to `0.05` (95% intervals).
- `name_lower`: column name for the lower bound in `partable`. Defaults to `:ci_lower`.
- `name_upper`: column name for the upper bound in `partable`. Defaults to `:ci_upper`.

# Returns
- a `Dict` with keys `name_lower` and `name_upper`, each mapping to a vector of bounds 
  over all parameters.
"""

@doc "$(_doc_normal_CI)"
function normal_CI(
        fitted, se; α = 0.05, name_lower = :ci_lower, name_upper = :ci_upper)
    qnt = quantile(Normal(0, 1), 1-α/2);
    sol = solution(fitted)
    return Dict(name_lower => sol - qnt*se, name_upper => sol + qnt*se)
end

@doc "$(_doc_normal_CI)"
function normal_CI!(
        partable,
        fitted,
        se;
        α = 0.05,
        name_lower = :ci_lower,
        name_upper = :ci_upper)
    cis = normal_CI(
        fitted, se; α, name_lower, name_upper)
    update_partable!(partable, name_lower, fitted, cis[name_lower])
    update_partable!(partable, name_upper, fitted, cis[name_upper])
    return cis
end
