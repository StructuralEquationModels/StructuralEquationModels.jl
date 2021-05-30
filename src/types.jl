## loss
abstract type LossFunction <: Function end

struct Loss{F <: Array{T} where {T <: LossFunction}}
    functions::F
end

## Diff
abstract type SemDiff end

abstract type DiffFunction <: Function end

struct SemAnalyticDiff{F <: Tuple} <: SemDiff
    algorithm
    options
    functions::F
end

## Obs
abstract type SemObs end

## Imply
abstract type Imply end
abstract type ImplyDefinition <: Imply end

## Interaction
abstract type SemInteraction end

## SEModel
abstract type AbstractSem end

struct Sem{O <: SemObs, I <: Imply, L <: Loss, D <: SemDiff} <: AbstractSem
    observed::O
    imply::I 
    loss::L 
    diff::D
end

struct CollectionSem{
    O <: Vector{O} where O <: SemObs,
    I <: Vector{I} where I <: Imply,
    L <: Vector{L} where L <: Loss,
    D <: Vector{D} where D <: SemDiff} <: AbstractSem
    observed_vec::O
    imply_vec::I
    loss_vec::L
    diff_vec::D
end

struct MGSem{V <: Vector{AS} where {
    AS <: AbstractSem}, D <: Vector{Vector{Int64}}} <: AbstractSem
    sem_vec::V
    par_subsets::D
end

struct FIMLSem{} <: AbstractSem
end
