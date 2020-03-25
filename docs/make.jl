using Documenter, sem

makedocs(;
    modules=[sem],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/Maximilian-Stefan-Ernst/sem.jl/blob/{commit}{path}#L{line}",
    sitename="sem.jl",
    authors="Maximilian Ernst, Aaron Peikert",
    assets=String[],
)

deploydocs(;
    repo="github.com/Maximilian-Stefan-Ernst/sem.jl",
)
