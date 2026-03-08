"""
    CFI(fit::SemFit)

Return the CFI.
"""
function CFI end


# if the user provides a baseline model
CFI(fit::SemFit, fit_baseline::SemFit) = 
    CFI(χ²(fit), dof(fit), χ²(fit_baseline), dof(fit_baseline))

# no baseline -> variance only model
CFI(fit::SemFit) = CFI(fit, fit.model)

function CFI(fit::SemFit, model::AbstractSem)
    dof₀ = dof_varonly(model)
    χ²₀ = χ²_varonly(model)
    return CFI(χ²(fit), dof(fit), χ²₀, dof₀)
end

# basic CFI function
function CFI(χ², dof, χ²₀, dof₀)
    λ = χ² - dof
    λ₀ = χ²₀ - dof₀
    return 1 - maximum([λ, 0])/maximum([λ, λ₀, 0])
end

###
function χ²_varonly(model::AbstractSemSingle)
    check_single_lossfun(model; throw_error = true)
    return χ²_varonly(model.loss.functions[1], model)
end

function χ²_varonly(model::SemEnsemble)
    check_single_lossfun(model; throw_error = true)
    return sum(χ²_varonly, model.sems)
end

function χ²_varonly(::SemML, model::AbstractSemSingle)
    N⁻ = (nsamples(model) - 1)
    S = obs_cov(observed(model))
    Σ₀ = Diagonal(S)
    p = nobserved_vars(model)
    return N⁻*(logdet(Σ₀) + tr(inv(Σ₀)*S) - logdet(S) - p)
end

# for the optimal variance only model, we have to solve 1/2 tr((I-XS⁻¹)^2) with X diagonal
function χ²_varonly(::SemWLS, model)
    N⁻ = (nsamples(model) - 1)
    S⁻¹ = inv((obs_cov(observed(model))))
    Σ₀ = Diagonal(inv(S⁻¹ .* S⁻¹)*diag(S⁻¹))
    return N⁻*0.5*tr((I - Σ₀*S⁻¹)^2)
end

# For FIML, an explicit bl model has to be passed
function χ²_varonly(::SemFIML, model)
    """
    Computing the CFI with FIML requires explicitely passing a fitted baseline model as
        CFI(fit::SemFit, fit_baseline::SemFit)
    """ |> ArgumentError |> throw
end

function dof_varonly(model::AbstractSemSingle)
    nparams_varonly = nobserved_vars(model)
    if MeanStruct(model.implied) === HasMeanStruct
        nparams_varonly *= 2
    end
    return n_dp(model) - nparams_varonly
end

dof_varonly(model::SemEnsemble) = sum(dof_varonly, model.sems)