function (loss::SemLossFunction)(par, model)
    F = zero(eltype(par))
    for i = 1:length(loss.functions)
        F += loss.functions[i](par, model)
        # all functions have to have those arguments??
    end
    return F
end

function (loss::SemLoss)(par, model)
    F = zero(eltype(par))
    for lossfun in loss.functions
        F += lossfun(par, model)
    end
    return F
end

function (loss::SemLoss)(par, prealloc, model)
    for lossfun in loss.functions lossfun(par, prealloc, model) end
end

function (loss::SemLoss)(par, F, G, H, model)
    loss = model.loss
    for lossfun in loss.functions
        lossfun.C(par, model)
    end

    if H != nothing
        
    end

    if G != nothing
        # code to compute gradient here
        # writing the result to the vector G
    end
    
    if F != nothing
        # value = ... code to compute objective function
        return value
    end
end

function (semml::SemML)(F, G, H, model)
    semml.C(model)
    if !isnothing(F)
        semml.F(model)
    end
    if !isnothing(G)
        semml.G(G, model)
    end
    if !isnothing(H)
        semml.H(H, model)
    end
end
