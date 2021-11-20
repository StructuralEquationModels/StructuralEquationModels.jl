struct SemDiffNLopt{A, B} <: SemDiff
    algorithm::A
    options::B
end