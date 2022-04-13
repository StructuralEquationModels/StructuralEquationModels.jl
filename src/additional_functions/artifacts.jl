# load an example data set
function example_data(name)
    basepath = @artifact_str name
    path = readdir(basepath, join = true)[1]
    data, header = readdlm(path, ',', header = true)
    data_raw = DataFrame(data, vec(header))
    data = identity.(data_raw)
    return data
end