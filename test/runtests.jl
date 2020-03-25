using sem
using Test

function ram(x)
      S =   [x[1] 0 0 0
            0 x[2] 0 0
            0 0 x[3] 0
            0 0 0 x[4]]

      F =  [1 0 0 0
            0 1 0 0
            0 0 1 0]

      A =  [0 0 0 1
            0 0 0 x[5]
            0 0 0 x[6]
            0 0 0 0]

      return (S, F, A)
end


x0 = append!([0.5, 0.5, 0.5, 0.5], ones(2))

@testset "sem.jl" begin
    @test x0 == [0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
end
