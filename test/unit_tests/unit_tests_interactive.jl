# requires: TestEnv to be installed globally, and the StructuralEquationModels package `]dev`ed
# example: julia test/unit_tests/unit_tests_interactive.jl matrix_helpers

try
    import TestEnv
    TestEnv.activate("StructuralEquationModels")
catch e
    @warn "Error initializing Test Env" exception=(e, catch_backtrace())
end
include("unit_tests.jl")
