############################################################################
### bootstrap based standard errors
############################################################################

function se_bootstrap(semfit::SemFit; n_boot = 3000, data = nothing, kwargs...)

    if isnothing(data)
        data = get_data(observed(model(semfit)))
    end

    data = prepare_data_bootstrap(data)

    start = solution(semfit)

    new_solution = zero(start)
    sum = zero(start)
    squared_sum = zero(start)

    n_failed = 0.0

    converged = true

    for _ = 1:n_boot

        sample_data = bootstrap_sample(data)
        new_model = swap_observed(model(semfit); data = sample_data, kwargs...)

        new_solution .= 0.0

        try
            new_solution = get_solution(sem_fit(new_model; start_val = start))
        catch
            n_failed += 1
        end

        @. sum += new_solution
        @. squared_sum += new_solution^2

        converged = true
    end

    n_conv = n_boot - n_failed
    sd = sqrt.(squared_sum/n_conv - (sum/n_conv).^2)
    print(n_failed, "\n")
    return sd

end

function prepare_data_bootstrap(data)
    return Matrix(data)
end

function bootstrap_sample(data)
    nobs = size(data, 1)
    index_new = rand(1:nobs, nobs)
    data_new = data[index_new, :]
    return data_new
end