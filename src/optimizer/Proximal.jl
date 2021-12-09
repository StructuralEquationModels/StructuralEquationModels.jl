function sem_fit(model::Sem{O, I, L, D}) where {O, I, L, D <: SemDiffProximal}

    solver = model.diff.algorithm
    minimizer, num_it = solver(model.imply.start_val, f = model, g = model.diff.g)

    return minimizer, num_it
end