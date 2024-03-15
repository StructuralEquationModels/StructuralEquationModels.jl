using Test

if haskey(ENV, "JULIA_ON_CI")
    @testset "multithreading_enabled" begin
        @test Threads.nthreads() >= 8
    end
end