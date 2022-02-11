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
        observed, 
        imply, 
        loss, 
        diff,
        kwargs...)

    kwargs = Dict{Symbol, Any}(kwargs...)

    if !isa(observed, SemObs)
        observed = observed(;kwargs...)
    end

    kwargs[:observed] = observed

    if !isa(imply, SemImply)
        imply = imply(;kwargs...)
    end

    kwargs[:imply] = imply

    loss_out = []

    for lossfun in loss
        if isa(lossfun, SemLossFunction)
            append!(loss_out, [lossfun])
        else
            lossfun = lossfun(;kwargs...)
            append!(loss_out, lossfun)
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

Sem(;
    observed = SemObsCommon,
    imply = RAM,
    loss = (SemML,),
    diff = SemDiffOptim,
    data = Matrix{Float64}(dat),
    start_val = start_val_ml)

Sem(;observed = SemObsCommon, imply = 1, loss = 1, diff = 1, data = Matrix{Float64}(dat), hi = 1)

loss_ml = SemML(semobserved, length(start_val_ml))

loss_ls = SemWLS(semobserved, length(start_val_ml))

loss = (loss_ml, loss_ls, )

typeof(loss)

loss_out = Vector()

for lossfun in loss
    if isa(lossfun, SemLossFunction)
        append!(loss_out, [lossfun])
    else
        lossfun = lossfun(;kwargs...)
        append!(loss_out, lossfun)
    end
end

lt = (loss_out...,)

#### try out
function tf(;a, b) return a*b end

function tf_outer(;kwargs...)
    b = 5
    kwargs = Dict(kwargs..., :b => b)
    tf(;kwargs...)
end


tf_outer(;c = 10)

