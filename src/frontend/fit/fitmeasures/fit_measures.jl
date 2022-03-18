fit_measures(sem_fit::SemFit{Mi, So, St, Mo, O} where {Mi, So, St, Mo <: AbstractSemSingle, O}) = 
    fit_measures(
        sem_fit,
        sem_fit.model.observed,
        sem_fit.model.imply,
        sem_fit.model.diff,
        sem_fit.model.loss.functions...
        )

fit_measures(sem_fit, obs, imp, diff, loss::Union{SemML, SemFIML}) = 
    fit_measures(
        sem_fit,
        npar,
        df,
        AIC,
        BIC,
        RMSEA,
        χ²,
        p_value,
        minus2ll,
        )

fit_measures(sem_fit, obs, imp, diff, loss::SemWLS) = 
    fit_measures(
        sem_fit,
        npar,
        df,
        RMSEA,
        χ²,
        p_value
        )

function fit_measures(sem_fit, args...)

    measures = Dict{Symbol, Float64}()
    
    for arg in args
        push!(measures, Symbol(arg) => arg(sem_fit))
    end

    return measures
end