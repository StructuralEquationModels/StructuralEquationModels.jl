using Test

@testset "multithreading_enabled" begin
    @test Threads.nthreads() == 8
end