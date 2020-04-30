abstract type Imply end

struct ImplyCommon{} <: Imply
    implied
end

struct ImplySparse{} <: Imply
    implied
end

struct ImplyDense{} <: Imply
    implied
end

struct ImplySymbolic{} <: Imply
    implied
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
