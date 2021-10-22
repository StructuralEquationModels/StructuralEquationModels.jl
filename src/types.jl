## loss
struct SemLossFunction{c, f, g, h}
    C::c
    F::f
    G::g
    H::h
end

struct SemLoss{F <: Tuple{T} where {T <: SemLossFunction}}
    functions::F
end

## Diff
abstract type SemDiff end

## Obs
abstract type SemObs end

## Imply
abstract type SemImply end

struct Sem{O <: SemObs, I <: SemImply, L <: SemLoss, D <: SemDiff}
    observed::O
    imply::I 
    loss::L 
    diff::D
end