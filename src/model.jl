abstract type SemObjective end

struct ram{T}
    S::T
    F::T
    A::T
end

mutable struct model{
        RAM <: ram,
        RAMF <: Function,
        OBS,
        PAR <: AbstractVecOrMat,
        OBJ <: SemObjective,
        OPT <: Optim.AbstractOptimizer}
    ram::RAM
    ramf::RAMF
    obs::OBS
    par::PAR
    objective::OBJ
    optimizer::OPT
    imp_cov
    optimizer_result
    par_uncertainty
    fitmeasure
end

struct SemObs{D, C, M}
    data::D
    cov::C
    mean::M
end

struct SemCalcCov end
struct SemCalcMean end

function SemObs(data; cov = SemCalcCov(), mean = SemCalcMean())
    SemObs(data, cov, mean)
end
import DataFrames.DataFrame
function SemObs(data::DataFrame; cov = SemCalcCov(), mean = SemCalcMean())
    data = convert(Matrix, data)
    SemObs(data, cov, mean)
end

function SemObs(data, cov::SemCalcCov, mean)
    cov = Statistics.cov(data)
    SemObs(data, cov, mean)
end

function SemObs(data, cov, mean::SemCalcMean)
    mean = vcat(Statistics.mean(data, dims = 1)...)
    SemObs(data, cov, mean)
end

function SemObs(data, cov::SemCalcCov, mean::SemCalcMean)
    cov = Statistics.cov(data)
    mean = vcat(Statistics.mean(data, dims = 1)...)
    SemObs(data, cov, mean)
end

function SemObs(; cov, mean = nothing)
    SemObs(nothing, cov, mean)
end

import Base.convert
convert(::Type{SemObs}, data) = SemObs(data)
convert(::Type{SemObs}, SemObs::SemObs) = SemObs

function model(ram, ramf, obs, par; objective = SemML(), optimizer = LBFGS())
    model(
    ram,
    ramf,
    convert(SemObs, obs),
    par,
    objective,
    optimizer,
    nothing,
    nothing,
    nothing,
    nothing)
end
