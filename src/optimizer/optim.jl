## connect do Optim.jl as backend
function sem_wrap_optim(par, F, G, H, sem::AbstractSem)
    if !isnothing(G) fill!(G, zero(eltype(G))) end
    if !isnothing(H) fill!(H, zero(eltype(H))) end
    return sem(par, F, G, H)
end

function sem_fit(model::Sem{O, I, L, D}) where {O, I, L, D <: SemDiffOptim}
    result = Optim.optimize(
                Optim.only_fgh!((F, G, H, par) -> sem_wrap_optim(par, F, G, H, model)),
                model.imply.start_val,
                model.diff.algorithm,
                autodiff = :forward,
                model.diff.options)
    return result
end