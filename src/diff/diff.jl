function SemAnalyticDiff(algorithm, options, functions)
    return SemAnalyticDiff(algorithm, options, functions, nothing)
end

function (diff::SemAnalyticDiff)(par, grad::AbstractVector, model)
    for i = 1:length(diff.functions)
        diff.functions[i](par, grad, model)
    end
end

function (diff::SemAnalyticDiff)(par, H::AbstractMatrix, model)
    for i = 1:length(diff.functions)
        diff.hessian_functions[i](par, H, model)
    end
end

function (model::Sem{A, B, C, D} 
    where {A, B, C, D <: SemAnalyticDiff})(par, grad)
    grad .= zero(eltype(grad))
    model.imply(par, model)
    model.diff(par, grad, model)
end