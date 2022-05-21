using Documenter, StructuralEquationModels

makedocs(
    sitename="StructuralEquationModels.jl",
    pages = [
        "index.md",
        "Tutorials" => [
            "A first model" => "tutorials/first_model.md",
            "Model specification" => [
                "tutorials/specification/specification.md",
                "tutorials/specification/graph_interface.md",
                "tutorials/specification/ram_matrices.md"],
            "Model construction" => [
                "tutorials/construction/construction.md",
                "tutorials/construction/outer_constructor.md",
                "tutorials/construction/build_by_parts.md"],
            "Model fitting" => "tutorials/fitting/fitting.md",
            "Model inspection" => [
                "tutorials/inspection/inspection.md"
            ],
            "Collections" => [
                "tutorials/collection/collection.md",
                "tutorials/collection/multigroup.md"
            ]
        ],
        "Developer documentation" => [
            "Extending the package" => "developer/extending.md",
            "Custom loss functions" => "developer/loss.md",
            "Custom imply types" => "developer/imply.md",
            "Custom diff types" => "developer/diff.md",
            "Custom observed types" => "developer/observed.md",
            "Custom sem types" => "developer/sem.md"
        ],
        "Performance tips" => [
            "Symbolic precomputation" => "performance/symbolic.md",
            "Starting values" => "performance/starting_values.md"
        ]
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = ["assets/favicon_zeta.ico"],
        ansicolor = true
    ),
    doctest = false
)

# doctest(StructuralEquationModels, fix=true)

deploydocs(
    repo = "github.com/StructuralEquationModels/StructuralEquationModels.jl",
    devbranch = "devel"
)