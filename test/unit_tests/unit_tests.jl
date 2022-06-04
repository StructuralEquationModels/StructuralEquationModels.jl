using Test, SafeTestsets

@safetestset "Multithreading" begin include("multithreading.jl") end

@safetestset "SemObs" begin include("data_input_formats.jl") end