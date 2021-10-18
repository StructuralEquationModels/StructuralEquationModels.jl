function (loss::Loss)(par, model)
    F = zero(eltype(par))
    for i = 1:length(loss.functions)
        F += loss.functions[i](par, model)
        # all functions have to have those arguments??
    end
    return F
end

# function (loss::Loss)(par, model, E, G)
#     if E != nothing
#         F = zero(eltype(model.imply.imp_cov))
#         for i = 1:length(loss.functions)
#             F += loss.functions[i](par, model, E, G)
#         end
#         return F
#     end
#     if G != nothing
#         G .= zero(eltype(G))
#         for i = 1:length(loss.functions)
#             loss.functions[i](par, model, E, G)
#         end
#         for i = 1:length(loss.functions)
#             G .+= loss.functions[i].grad
#         end
#
#     end
# end

function Sem_fgh!(F, G, H, model)
    loss = model.loss
    if F != nothing
    for i = 1:size(loss.functions)

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

# function (loss::Loss)(par, model, E, G)
#
#     #common computations with 2 arguments
#     #store computations in fields
#     for i = 1:length(loss.functions)
#         loss.functions[i](par, model)
#     end
#
#     for i = 1:length(loss.functions)
#         loss.functions[i](par, model, E, G)
#     end
#
#     if E != nothing
#         objective = zero(eltype(model.imply.imp_cov))
#         for i = 1:length(loss.functions)
#             objective += loss.functions[i].objective[1]
#         end
#         return objective
#     end
#
#     if G != nothing
#
#         G .= zero(eltype(G))
#         for i = 1:length(loss.functions)
#             G .+= loss.functions[i].grad
#         end
#
#     end
# end


