abstract type SemDiff end

struct SemForwardDiff <: SemDiff
    options #Optim.Options() call fot optimize()
end

struct SemReverseDiff{} <: SemDiff
    # For preallocations, see the examples in ReverseDiff
end

struct SemFiniteDiff{} <: SemDiff
    # For preallocations, see the examples in ReverseDiff
end
