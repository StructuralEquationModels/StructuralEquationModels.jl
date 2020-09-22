#imply is the new ram
abstract type Imply end

struct ImplyCommon{A <: AbstractArray} <: Imply
    implied::A
end

struct ImplySparse{A} <: Imply
    implied::A
end

struct ImplyDense{A <: Array{Float64}} <: Imply
    implied::A
end

struct ImplySymbolic{A} <: Imply
    implied::A
end

function (imply::ImplySparse)(par)
    imply.implied .=
end

function (imply::ImplyDense)(par)
    imply.implied .=
end

function (imply::ImplySymbolic)(par)
    imply.implied .=
end

function (imply::ImplyCommon)(par)
    imply.implied .=
end
