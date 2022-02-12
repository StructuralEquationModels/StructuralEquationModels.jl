struct SemDiffOptim{A, B} <: SemDiff
    algorithm::A
    options::B
end

SemDiffOptim(;algorithm = LBFGS(), options = Optim.Options(;f_tol = 1e-10, x_tol = 1.5e-8), kwargs...) = SemDiffOptim(algorithm, options)