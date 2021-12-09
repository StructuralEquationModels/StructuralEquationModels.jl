struct SemDiffProximal{A, B, C} <: SemDiff
    algorithm::A
    options::B
    g::C
end