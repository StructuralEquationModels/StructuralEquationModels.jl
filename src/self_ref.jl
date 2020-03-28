struct lazy{T}
    data_raw::AbstractArray{T}
    data::Ref{Union{Nothing, AbstractArray{T}}}
    lazy(data_raw::AbstractArray{T}) where T = new{T}(data_raw, nothing)
end

function data(model::lazy)
    isnothing(model.data[]) && (model.data[] = [sum(model.data_raw*3)])
    return model.data[]
end

test = lazy([1 2 3])
test.data[]
data(test)
test.data[]
