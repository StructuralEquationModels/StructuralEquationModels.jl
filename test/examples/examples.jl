using Test, SafeTestsets

@safetestset "Political Democracy" begin include("political_democracy/political_democracy.jl") end
#@safetestset "Political Democracy Parser" begin include("political_democracy_parser.jl") end
#@safetestset "Political Democracy NLopt" begin include("political_democracy_constructor_NLopt.jl") end
#@safetestset "Recover Parameters" begin include("recover_parameters_twofact.jl") end
#@safetestset "Multigroup" begin include("multigroup.jl") end
#@safetestset "Multigroup Parser" begin include("multigroup_parser.jl") end