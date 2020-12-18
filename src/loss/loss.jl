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

function (loss::Loss)(par, model, E, G)

    if E != nothing

        for i = 1:length(loss.functions)
            loss.functions[i](par, model, E, G)
        end

        objective = zero(eltype(model.imply.imp_cov))
        for i = 1:length(loss.functions)
            objective += loss.functions[i].objective[1]
        end

        return objective
    end

    if G != nothing

        for i = 1:length(loss.functions)
            loss.functions[i](par, model, E, G)
        end

        G .= zero(eltype(G))
        for i = 1:length(loss.functions)
            G .+= loss.functions[i].grad
        end

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


