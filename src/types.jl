## loss
abstract type LossFunction <: Function end

struct Loss{F <: Array{T} where {T <: LossFunction}}
    functions::F
end

## Diff
abstract type SemDiff end

## Obs
abstract type SemObs end

## Imply
abstract type Imply end


## SEModel
struct Sem{O <: SemObs, I <: Imply, L <: Loss, D <: SemDiff}
    observed::O
    imply::I # former ram
    loss::L # list of loss functions
    diff::D
end
