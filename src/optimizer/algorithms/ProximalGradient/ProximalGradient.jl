function proximalgradient(model, g; tolerance = 1e-6, maxit_linesearch = 50, maxit = 300)
    x = copy(model.imply.start_val)
    ∇f = similar(x)
    x_new = similar(x)

    λ = 1.0
    β = 0.5

    converged = false
    iterations = 0

    for _ in 1:maxit
        λ = 1.0

        f_x = objective_gradient!(∇f, model, x)

        ls_success = false

        for _ in 1:maxit_linesearch
            x_1, _ = prox(g, x - λ*∇f, λ)
            x_new .= x_1
            
            #tol = 10 * eps(Float64) * (1 + abs(f_x))
            
            if objective!(model, x_new) <= f_x
                ls_success = true
                break
            end

            λ *= β
        end

        if !ls_success
            @warn "linesearch failed"
            break
        end
        #x_new
        if norm(x-x_new, Inf)/λ < tolerance
            converged = true
            break
        end

        copyto!(x, x_new)

        iterations += 1

    end

    return x_new, converged, iterations

end

#= function f̂(f_x, ∇f_x, x, x_new, λ)
    x_diff = x_new - x
    return f_x + dot(∇f_x, x_diff) + 0.5*λ*norm(x_diff)
end =#