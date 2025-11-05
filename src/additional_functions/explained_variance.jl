
"""
explained_variance(fitted)

Compute explained variance (R²) for each observed variable in a fitted SEM.

Returns a DataFrame with:
- :var         — observed variable name (Symbol)
- :r2_implied  — 1 - ( residual variance / model-implied variance)

"""
function explained_variance(fitted::SemFit)
# 1) Observed variable order as used by the model
obs = Symbol.(fitted.model.observed.observed_vars)

# 2) Model-implied covariance (diagonal are model-implied total variances)
Σ_imp = fitted.model.implied.Σ
var_imp = diag(Σ_imp)

# 3) Get model-implied variances of observed variables
resvars = diag(fitted.model.implied.F*fitted.model.implied.S*transpose(fitted.model.implied.F))

# 4) Compute R² (model-implied)
r2_imp = 1 .- (resvars ./ var_imp)

# prepare return object
df = DataFrame(var = obs, r2_implied = r2_imp)
return df

end
