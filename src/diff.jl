struct SemForwardDiff <: SemDiff
    algorithm
    options #Optim.Options() call to optimize()
end

# function SemForwardDiff()
#     return SemForwardDiff(Optim.Options())
# end

struct SemReverseDiff{} <: SemDiff
    algorithm
    options
    # For preallocations, see the examples in ReverseDiff
end

struct SemFiniteDiff{} <: SemDiff
    algorithm
    options
end

# function SemFiniteDiff()
#     return SemFiniteDiff(Optim.Options())
# end
