struct SemDiffOptim{A, B} <: SemDiff
    algorithm::A
    options::B
end