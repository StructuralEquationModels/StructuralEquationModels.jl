struct SemDiffNLopt{A, B} <: SemDiff
    algorithm::A
    options::B
end

SemDiffNLopt(;algorithm = :LD_LBFGS, options = nothing, kwargs...) = SemDiffNLopt(algorithm, options)