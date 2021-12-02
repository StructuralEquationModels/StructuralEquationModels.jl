function test_gradient(model, parameters)
    true_grad = FiniteDiff.finite_difference_gradient(x -> objective!(model, x))

    # F and G
    gradient(model) .= 0
    model(parameters, 1.0, similar(parameters), nothing)
    correct1 = isapprox(gradient(model), true_grad)

    # only G
    gradient(model) .= 0
    model(parameters, nothing, similar(parameters), nothing)
    correct2 = isapprox(gradient(model), true_grad)

    return correct1 & correct2
end

function test_hessian(model, parameters)
    H = one(length(parameters), length(parameters))
    true_hessian = FiniteDiff.finite_difference_hessian(x -> objective!(model, x))

    # F and H
    hessian(model) .= 0
    model(parameters, 1.0, nothing, H)
    correct1 = isapprox(hessian(model), true_hessian; rtol = 1/1000)

    # G and H
    hessian(model) .= 0
    model(parameters, nothing, similar(parameters), H)
    correct2 = isapprox(hessian(model), true_hessian; rtol = 1/1000)

    # only H
    hessian(model) .= 0
    model(parameters, nothing, nothing, H)
    correct3 = isapprox(hessian(model), true_hessian; rtol = 1/1000)
    
    return correct1 & correct2 & correct3
end