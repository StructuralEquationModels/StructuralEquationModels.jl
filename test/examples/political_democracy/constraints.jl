# NLopt constraints -----------------------------------------------

# 1.5*x1 == x2 (aka 1.5*x1 - x2 == 0)
#= function eq_constraint(x, grad)
    if length(grad) > 0
        grad .= 0.0
        grad[1] = 1.5
        grad[2] = -1.0
    end
    1.5*x[1] - x[2]
end =#

# x30 ≥ 1.0 (aka 1.0 - x30 ≤ 0)
#= function ineq_constraint(x, grad)
    if length(grad) > 0
        grad .= 0.0
        grad[30] = -1.0
    end
    1.0 - x[30]
end =#

# x30*x31 ≥ 0.6 (aka 0.6 - x30*x31 ≤ 0)
function ineq_constraint(x, grad)
    if length(grad) > 0
        grad .= 0.0
        grad[30] = -x[31]
        grad[31] = -x[30]
    end
    0.6 - x[30]*x[31]
end

constrained_diff = SemDiffNLopt(;
    algorithm = :AUGLAG,
    local_algorithm = :LD_LBFGS,
    local_options = Dict(
        :ftol_rel => 1e-6
    ),
    # equality_constraints = NLoptConstraint(;f = eq_constraint, tol = 1e-14),
    inequality_constraints = NLoptConstraint(;f = ineq_constraint, tol = 0.0),
)

model_ml_constrained = Sem(
    specification = spec,
    data = dat,
    diff = constrained_diff
)

solution_constrained = sem_fit(model_ml_constrained)

# NLopt option setting --------------------------------------------

model_ml_maxeval = Sem(
    specification = spec,
    data = dat,
    diff = SemDiffNLopt,
    options = Dict(:maxeval => 10)
)

############################################################################
### test solution
############################################################################

@testset "ml_solution_maxeval" begin
    solution_maxeval = sem_fit(model_ml_maxeval)
    @test solution_maxeval.optimization_result.problem.numevals == 10
    @test solution_maxeval.optimization_result.result[3] == :MAXEVAL_REACHED
end

@testset "ml_solution_constrained" begin
    solution_constrained = sem_fit(model_ml_constrained)
    @test solution_constrained.solution[31]*solution_constrained.solution[30] >= 0.6
    @test all(abs.(solution_constrained.solution) .< 10)
    @test_skip solution_constrained.optimization_result.result[3] == :FTOL_REACHED
    @test abs(solution_constrained.minimum - 21.21) < 0.01
end