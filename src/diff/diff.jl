struct SemForwardDiff <: SemDiff
    algorithm
    options #Optim.Options() call to optimize()
end

# function SemForwardDiff()
#     return SemForwardDiff(Optim.Options())
# end

struct SemReverseDiff{} <: SemDiff
    algorithm
    options
    # For preallocations, see the examples in ReverseDiff
end

struct SemFiniteDiff{} <: SemDiff
    algorithm
    options
end

# function SemFiniteDiff()
#     return SemFiniteDiff(Optim.Options())
# end


function (diff::SemAnalyticDiff)(par, grad, model)
    for i = 1:length(diff.functions)
        diff.functions[i](par, grad, model)
        # all functions have to have those arguments??
    end
end

function (model::Sem{A, B, C, D} 
    where {A, B, C, D <: SemAnalyticDiff})(par, grad)
    model.imply(par, model)
    model.diff(par, grad, model)
end