# load an example data set
function example_data(name)
    #print(@artifact_str name)
    #hash = artifact_hash(name, joinpath(replace(@__DIR__, "additional_functions" => ""), "Artifacts.toml"))
    #basepath = artifact_path(hash)
    basepath = @artifact_str name
    path = readdir(basepath, join = true)[1]
    data, header = readdlm(path, ',', header = true)
    data_raw = DataFrame(data, vec(header))
    data = identity.(data_raw)
    return data
end