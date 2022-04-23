using Test, SafeTestsets

@safetestset "Recover Parameters" begin include("recover_parameters_twofact.jl") end
@safetestset "Multigroup" begin include("multigroup.jl") end
@safetestset "Multigroup Parser" begin include("multigroup_parser.jl") end