# load an example data set
function example_data(name)
    basepath = @artifact_str name
    paths = readdir(basepath)
    if length(paths) == 1
        path = joinpath(basepath, paths[1])
        data, header = readdlm(path, ',', header = true)
        data_raw = DataFrame(data, vec(header))
        data = identity.(data_raw)
        return data
    else
        out = Dict()
        for path in paths
            key = Symbol(split(path, ".")[1])
            path = joinpath(basepath, path)
            data, header = readdlm(path, ',', header = true)
            data_raw = DataFrame(data, vec(header))
            data = identity.(data_raw)
            push!(out, key => data)
        end
        return out
    end
end