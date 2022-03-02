function fit_measures(sem_fit)

    measures = Dict{Symbol, Float64}()
    
    push!(measures, :F => Fₘᵢₙ(sem_fit))
    push!(measures, :p => p_value(sem_fit))
    push!(measures, :χ² => χ²(sem_fit))

    return measures
end