using Test, SafeTestsets

@safetestset "Multithreading" begin
    include("multithreading.jl")
end

@safetestset "Matrix algebra helper functions" begin
    include("matrix_helpers.jl")
end

@safetestset "SemObserved" begin
    include("data_input_formats.jl")
end
