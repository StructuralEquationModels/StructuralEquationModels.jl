using Pkg

Pkg.add(PackageSpec(; url="https://github.com/StructuralEquationModels/StructuralEquationModels.jl", rev = "devel"))

Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Symbolics") 
Pkg.add("Optim")
Pkg.add("LineSearches")
Pkg.add("BenchmarkTools")

Pkg.test("StructuralEquationModels")