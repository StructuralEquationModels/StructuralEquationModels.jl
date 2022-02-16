##############################################################
# constructor for Sem types
##############################################################

function Sem(;
        observed::O = SemObsCommon,
        imply::I = RAM,
        loss::L = (SemML,),
        diff::D = SemDiffOptim,
        kwargs...) where {O, I, L, D}

    kwargs = Dict{Symbol, Any}(kwargs...)

    kwargs[:observed_type] = O <: Type ? observed : typeof(observed)
    kwargs[:imply_type] = I <: Type ? imply : typeof(imply)
    kwargs[:loss_types] = [lossfun isa SemLossFunction ? typeof(lossfun) : lossfun for lossfun in loss]
    kwargs[:diff_type] = D <: Type ? diff : typeof(diff)

    if O <: Type
        observed = observed(;kwargs...)
    end

    kwargs[:observed] = observed

    if !isa(imply, SemImply)
        imply = imply(;kwargs...)
    end

    kwargs[:imply] = imply
    kwargs[:n_par] = length(imply.start_val)

    loss_out = []

    for lossfun in loss
        if isa(lossfun, SemLossFunction)
            append!(loss_out, [lossfun])
        else
            lossfun = lossfun(;kwargs...)
            push!(loss_out, lossfun)
        end
    end

    loss = SemLoss((loss_out...,))

    kwargs[:loss] = loss

    if !isa(diff, SemDiff)
        diff = diff(;kwargs...)
    end

    sem = Sem(observed, imply, loss, diff)

    return sem

end

function SemFiniteDiff(;
        observed::O = SemObsCommon,
        imply::I = RAM,
        loss::L = (SemML,),
        diff::D = SemDiffOptim,
        has_gradient = false,
        kwargs...) where {O, I, L, D}

    kwargs = Dict{Symbol, Any}(kwargs...)

    kwargs[:observed_type] = O <: Type ? observed : typeof(observed)
    kwargs[:imply_type] = I <: Type ? imply : typeof(imply)
    kwargs[:loss_types] = [lossfun isa SemLossFunction ? typeof(lossfun) : lossfun for lossfun in loss]
    kwargs[:diff_type] = D <: Type ? diff : typeof(diff)

    if O <: Type
        observed = observed(;kwargs...)
    end

    kwargs[:observed] = observed

    if !isa(imply, SemImply)
        imply = imply(;kwargs...)
    end

    kwargs[:imply] = imply
    kwargs[:n_par] = length(imply.start_val)

    loss_out = []

    for lossfun in loss
        if isa(lossfun, SemLossFunction)
            append!(loss_out, [lossfun])
        else
            lossfun = lossfun(;kwargs...)
            push!(loss_out, lossfun)
        end
    end

    loss = SemLoss((loss_out...,))

    kwargs[:loss] = loss

    if !isa(diff, SemDiff)
        diff = diff(;kwargs...)
    end

    sem = SemFiniteDiff(observed, imply, loss, diff, has_gradient)

    return sem

end

function SemForwardDiff(;
        observed::O = SemObsCommon,
        imply::I = RAM,
        loss::L = (SemML,),
        diff::D = SemDiffOptim,
        has_gradient = false,
        kwargs...) where {O, I, L, D}

    kwargs = Dict{Symbol, Any}(kwargs...)

    kwargs[:observed_type] = O <: Type ? observed : typeof(observed)
    kwargs[:imply_type] = I <: Type ? imply : typeof(imply)
    kwargs[:loss_types] = [lossfun isa SemLossFunction ? typeof(lossfun) : lossfun for lossfun in loss]
    kwargs[:diff_type] = D <: Type ? diff : typeof(diff)

    if O <: Type
        observed = observed(;kwargs...)
    end

    kwargs[:observed] = observed

    if !isa(imply, SemImply)
        imply = imply(;kwargs...)
    end

    kwargs[:imply] = imply
    kwargs[:n_par] = length(imply.start_val)

    loss_out = []

    for lossfun in loss
        if isa(lossfun, SemLossFunction)
            append!(loss_out, [lossfun])
        else
            lossfun = lossfun(;kwargs...)
            push!(loss_out, lossfun)
        end
    end

    loss = SemLoss((loss_out...,))

    kwargs[:loss] = loss

    if !isa(diff, SemDiff)
        diff = diff(;kwargs...)
    end

    sem = SemForwardDiff(observed, imply, loss, diff, has_gradient)

    return sem

end

##############################################################
# pretty printing
##############################################################

#= function Base.show(io::IO, sem::Sem{O, I, L, D})  where {O, I, L, D}
    lossfuntypes = @. nameof(typeof(sem.loss.functions))
    print(io, "Sem{$(nameof(O)), $(nameof(I)), $lossfuntypes, $(nameof(D))}")
end =#

function Base.show(io::IO, sem::Sem{O, I, L, D})  where {O, I, L, D}
    lossfuntypes = @. string(nameof(typeof(sem.loss.functions)))
    lossfuntypes = "   ".*lossfuntypes.*("\n")
    print(io, "Structural Equation Model \n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   observed:  $(nameof(O)) \n")
    print(io, "   imply:     $(nameof(I)) \n")
    print(io, "   diff:      $(nameof(D)) \n")
end

function Base.show(io::IO, sem::Sem{O, I, L, D})  where {O, I, L, D}
    lossfuntypes = @. string(nameof(typeof(sem.loss.functions)))
    lossfuntypes = "   ".*lossfuntypes.*("\n")
    print(io, "Structural Equation Model \n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   observed:  $(nameof(O)) \n")
    print(io, "   imply:     $(nameof(I)) \n")
    print(io, "   diff:      $(nameof(D)) \n")
end

function Base.show(io::IO, sem::SemFiniteDiff{O, I, L, D})  where {O, I, L, D}
    lossfuntypes = @. string(nameof(typeof(sem.loss.functions)))
    lossfuntypes = "   ".*lossfuntypes.*("\n")
    print(io, "Structural Equation Model : Finite Diff Approximation\n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   observed:  $(nameof(O)) \n")
    print(io, "   imply:     $(nameof(I)) \n")
    print(io, "   diff:      $(nameof(D)) \n") 
end

function Base.show(io::IO, sem::SemForwardDiff{O, I, L, D})  where {O, I, L, D}
    lossfuntypes = @. string(nameof(typeof(sem.loss.functions)))
    lossfuntypes = "   ".*lossfuntypes.*("\n")
    print(io, "Structural Equation Model : Forward Mode Autodiff\n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   observed:  $(nameof(O)) \n")
    print(io, "   imply:     $(nameof(I)) \n")
    print(io, "   diff:      $(nameof(D)) \n") 
end

function Base.show(io::IO, loss::SemLoss)
    lossfuntypes = @. string(nameof(typeof(loss.functions)))
    lossfuntypes = "   ".*lossfuntypes.*("\n")
    print(io, "SemLoss \n")
    print(io, "- Loss Functions \n")
    print(io, lossfuntypes...)
    print(io, "- Fields \n")
    print(io, "   F:  $(typeof(loss.F))) \n")
    print(io, "   G:  $(typeof(loss.G))) \n")
    print(io, "   H:  $(typeof(loss.H))) \n") 
end

function Base.show(io::IO, models::SemEnsemble)

    print(io, "SemEnsemble \n")
    print(io, "- Number of Models: $(models.n) \n")
    print(io, "- Weights: $(round.(models.weights, digits = 3)) \n")

    print(io, "\n")
    print(io, "Fields \n")
    print(io, "   diff:  $(nameof(typeof(models.diff))) \n")
    print(io, "   start_val:  $(typeof(models.start_val)) \n")
    print(io, "   F:  $(typeof(models.F))) \n")
    print(io, "   G:  $(typeof(models.G))) \n")
    print(io, "   H:  $(typeof(models.H))) \n")

    print(io, "\n")
    print(io, "Models: \n")
    print(io, "=========================", "\n")
    for (model, i) in zip(models.sems, 1:models.n)
        print(io, "----------- ", i, " -----------", "\n")
        print(io, model)
    end
end