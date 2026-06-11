using Test, SafeTestsets

# Define available test sets
available_tests = Dict(
    "multithreading" => "Multithreading",
    "matrix_helpers" => "Matrix algebra helper functions",
    "data_input_formats" => "SemObserved",
    "specification" => "SemSpecification",
    "model" => "Sem model",
    "StatsAPI" => "StatsAPI",
)

# Determine which tests to run based on command-line arguments
selected_tests = isempty(ARGS) ? collect(keys(available_tests)) : ARGS

@testset "All Tests" begin
    for file in selected_tests
        if haskey(available_tests, file)
            let file_ = file, test_name = available_tests[file]
                # Compute the literal values
                test_sym = Symbol(file_)
                file_jl = file_ * ".jl"
                # Build the expression with no free variables:
                ex = quote
                    @safetestset $(Symbol(test_sym)) = $test_name begin
                        include($file_jl)
                    end
                end
                eval(ex)
            end
        else
            @warn "Test file '$file' not found in available tests. Skipping."
        end
    end
end
