using Enzyme, BenchmarkTools, FiniteDiff, LinearAlgebra

f(x) = x^2

f(x) = x[1]*x[2]

inp =  [2.0 1.0
        0.0 1.0]
dinp = [0.0 0.0
        0.0 0.0]

f2(x) = inv(x)

@benchmark autodiff(det, Duplicated(inp, dinp))

@benchmark FiniteDi