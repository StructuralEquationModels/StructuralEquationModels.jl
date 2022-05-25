"""
    example_data(name)

Downloads and returns the `name` example dataset.
Available datasets are `"political_democracy"` and `"holzinger_swineford"` as well as
versions with missign values, `"political_democracy_missing"` and 
`"holzinger_swineford_missing"`.
"""
function example_data(name)
    basepath = @artifact_str name
    paths = readdir(basepath)
    if length(paths) == 1
        path = joinpath(basepath, paths[1])
        data, header = readdlm(path, ',', header = true)
        data_raw = DataFrame(data, vec(header))
        for col in eachcol(data_raw)
            replace!(col,"NA" => missing)
        end
        data = identity.(data_raw)
        return data
    else
        out = Dict()
        for path in paths
            key = Symbol(split(path, ".")[1])
            path = joinpath(basepath, path)
            data, header = readdlm(path, ',', header = true)
            data_raw = DataFrame(data, vec(header))
            for col in eachcol(data_raw)
                replace!(col,"NA" => missing)
            end
            data = identity.(data_raw)
            push!(out, key => data)
        end
        return out
    end
end