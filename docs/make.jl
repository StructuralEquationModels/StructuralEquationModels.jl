using Documenter, StructuralEquationModels

# push!(LOAD_PATH,"/home/maximilian/Documents/repositories/StructuralEquationModels.jl/src")

makedocs(
    sitename="StructuralEquationModels.jl",
    pages = [
        "index.md",
        "Tutorials" => [
            "A first model" => "tutorials/first_model.md",
            "Model specification" => [],
            "Model construction" => ["tutorials/construction/build_by_parts.md"],
            "Model fitting" => [],
            "Model inspection" => [],
            "Multigroup SEM" => [
                "tutorials/multigroup/multigroup.md",
                "tutorials/multigroup/collection.md"
            ]
        ],
        "Developer documentation" => [
            "Custom loss functions" => "developer/loss.md",
            "Custom imply types" => "developer/imply.md",
            "Custom sem types" => "developer/sem.md"
        ],
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    doctest = false
)

# doctest(StructuralEquationModels, fix=true)

deploydocs(
    repo = "github.com/StructuralEquationModels/StructuralEquationModels.jl",
    devbranch = "devel"
)