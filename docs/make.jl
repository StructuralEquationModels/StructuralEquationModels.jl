using Documenter, StructuralEquationModels

# push!(LOAD_PATH,"/home/maximilian/Documents/repositories/StructuralEquationModels.jl/src")

makedocs(
    sitename="StructuralEquationModels.jl",
    pages = [
        "index.md",
        "Tutorials" => [
            "Fit a SEM" => [
                "tutorials/high_level.md",
                "tutorials/build_by_parts.md"],
            "Multigroup SEM" => [
                "tutorials/multigroup.md",
                "tutorials/collection.md"
            ],
            "Custom loss functions" => "tutorials/loss.md",
            "Custom imply types" => "tutorials/imply.md",
            "Custom sem types" => "tutorials/sem.md"
        ],
    ]
)

deploydocs(
    repo = "github.com/StructuralEquationModels/StructuralEquationModels.jl.git",
)

function testfun(args...)
    print(args)
end