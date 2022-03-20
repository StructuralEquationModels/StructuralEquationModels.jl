############################################################################
### connect to NLopt.jl as backend
############################################################################

# wrapper to define the objective
function sem_wrap_nlopt(par, G, sem::AbstractSem)
    need_gradient = length(G) != 0
    sem(par, true, need_gradient, false)
    if need_gradient G .= gradient(sem) end
    return objective(sem)
end

# construct SemFit from fitted NLopt object
function SemFit_NLopt(optimization_result, model::AbstractSem, start_val, opt)
    return SemFit(
        optimization_result[1],
        optimization_result[2],
        start_val,
        model,
        Dict(:result => optimization_result, :problem => opt)
    )
end

# sem_fit method
function sem_fit(model::Sem{O, I, L, D}; kwargs...) where {O, I, L, D <: SemDiffNLopt}

    # starting values
    if !isa(start_val, Vector)
        start_val = start_val(model; kwargs...)
    end

    # construct the NLopt problem
    opt = construct_NLopt_problem(model.diff, start_val)
    set_NLopt_constraints!(opt, diff)   
    opt.min_objective = (par, G) -> sem_wrap_nlopt(par, G, model)

    # fit
    result = NLopt.optimize(opt, start_val)

    return SemFit_NLopt(result, model, start_val, opt)
end

############################################################################
### additional functions
############################################################################

function construct_NLopt_problem(diff::SemDiffNLopt, npar)
    opt = Opt(diff.algorithm, npar)

    for key in keys(diff.options)
        setproperty!(opt, key, diff.options[key])
    end

    return opt

end

function set_NLopt_constraints!(opt, diff::SemDiffNLopt)
    for con in diff.inequality_constraints
        inequality_constraint!(opt::Opt, con.f, con.tol)
    end
    for con in diff.equality_constraints
        equality_constraint!(opt::Opt, con.f, con.tol)
    end
end