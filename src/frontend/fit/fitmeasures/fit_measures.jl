fit_measures(sem_fit) = 
    fit_measures(
        sem_fit,
        n_par,
        df,
        AIC,
        BIC,
        RMSEA,
        χ²,
        p_value,
        minus2ll
        )

function fit_measures(sem_fit, args...)

    measures = Dict{Symbol, Float64}()
    
    for arg in args
        push!(measures, Symbol(arg) => arg(sem_fit))
    end

    return measures
end