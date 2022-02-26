using Test, SafeTestsets

@safetestset "Political Democracy" begin include("political_democracy_constructor.jl") end
@safetestset "Political Democracy Parser" begin include("political_democracy_parser.jl") end
@safetestset "Recover Parameters" begin include("recover_parameters_twofact.jl") end
@safetestset "Multigroup" begin include("multigroup.jl") end