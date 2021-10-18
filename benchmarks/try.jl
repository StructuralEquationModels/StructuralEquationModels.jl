using FiniteDiff

function f1(x)
    x[1]*x[2]/(x[1]+x[2])
end

function f2(x)
    x[1] + x[2]
end

x = rand(2)

f3(x) = f1(x)+f2(x)

FiniteDiff.finite_difference_hessian(f3, x) ≈ FiniteDiff.finite_difference_hessian(f1, x) + FiniteDiff.finite_difference_hessian(f2, x)


