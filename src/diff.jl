abstract type SemDiff end

struct SemDiffForward <: SemDiff
    options #Optim.Options() call fot optimize()
end

struct SemDiffReverse{} <: SemDiff
    # For preallocations, see the examples in ReverseDiff
end

struct SemDiffFinite{} <: SemDiff
    # For preallocations, see the examples in ReverseDiff
end
