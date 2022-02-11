##############################################################
# constructor for Sem types
##############################################################

Base.@kwdef struct RAMMatrices
    A
    S
    F
    M = nothing
    parameters
end

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