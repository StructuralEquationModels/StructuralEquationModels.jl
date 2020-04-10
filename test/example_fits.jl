include("../test/example_models.jl");

datas = (one_fact_dat, three_mean_dat, three_path_dat)
model_funcs = (one_fact_func, three_mean_func, three_path_func)
start_values = (
    vcat(fill(1, 4), fill(0.5, 2)),
    vcat(fill(1, 9), fill(1, 3), fill(0.5, 3), fill(0.5, 6)),#, vec(mean(convert(Matrix{Float64}, three_mean_dat), dims = 1))),
    vcat(fill(1, 14), fill(0.5, 17))
    )

optimizers = (LBFGS(), GradientDescent(), Newton())


for i in 1:length(datas)
    for j in 1:length(optimizers)
        model = sem.model(model_funcs[i],
            datas[i],
            start_values[i])
    end
end

@benchmark tr = ram(test.ram(test.par)[1],
    test.ram(test.par)[2],
    test.ram(test.par)[3])

tr

@benchmark three_mean_func(tr, zeros(21))


test = sem.model(model_funcs[1], datas[1], start_values[1])

Optim.minimizer(fit(test))

Optim.minimizer(fit(test))



### model 2


test = sem.model(ram(three_mean_func(start_values[2])[1],
                        three_mean_func(start_values[2])[2],
                        three_mean_func(start_values[2])[3]),
                ramfunc,
                datas[2],
                start_values[2])



function imp_cov_t(ram::ram{Array{Float64,2}})
      invia = LinearAlgebra.inv!(factorize(I - ram.A)) # invers of I(dentity) minus A matrix
      imp = ram.F*invia*factorize(ram.S)*transpose(invia)*transpose(ram.F)
      return imp
end

testram = ram(three_mean_func(start_values[2])[1],
                        three_mean_func(start_values[2])[2],
                        three_mean_func(start_values[2])[3])

imp_cov_t(testram)

invia = LinearAlgebra.inv!(factorize(I - testram.A))

@benchmark begin # invers of I(dentity) minus A matrix
    imp = testram.F*invia*testram.S*transpose(invia)*transpose(testram.F)
end

@benchmark test.objective(test.par, test)

fit(test)

A = [0.0 0.0
    0.0 0.0]

function parfunc(A, par)
    A[1,1] = par[1]
end

function est(A, par)
    parfunc(A, par)
    A[1,1]^2
end

objective = par -> est(A, par)

optimize(
    objective,
    [6.0],
    LBFGS(),
    autodiff = :forward)


invia = inv(I - test.ram(test.par)[3])

@benchmark ms[2]*invia*ms[1]*transpose(invia)*transpose(ms[2])

function imp_cov(D)
      invia = inv(I - D[3]) # invers of I(dentity) minus A matrix
      imp = D[2]*invia*D[1]*transpose(invia)*transpose(D[2])
      return imp
end

A = test.ram(test.par)[1]
B = test.ram(test.par)[2]
C = test.ram(test.par)[3]

D = [A, B, C]

@benchmark imp_cov(D)

@benchmark three_mean_func(start_values[2])

mat[3]


B = Array{Float64}(undef, 12, 12)

@benchmark begin
    B = rand(12, 12)
    B[:,:] = A
end

function allocate_matr(parameters, ram)
    A = Array{eltype}(undef, size())

@benchmark C = ram(rand(12,12), rand(12,12), rand(12,12))


B

A =             [0.0  0  0  0  0  0  0  0  0  1     0    0.0
                0  0  0  0  0  0  0  0  0  0.5    0     0
                0  0  0  0  0  0  0  0  0  0.5    0     0
                0  0  0  0  0  0  0  0  0  0     1      0
                0  0  0  0  0  0  0  0  0  0     0.5    0
                0  0  0  0  0  0  0  0  0  0     0.5    0
                0  0  0  0  0  0  0  0  0  0     0     1
                0  0  0  0  0  0  0  0  0  0     0     0.5
                0  0  0  0  0  0  0  0  0  0     0     0.5
                0  0  0  0  0  0  0  0  0  0     0     0
                0  0  0  0  0  0  0  0  0  0     0     0
                0  0  0  0  0  0  0  0  0  0     0     0]


setin

A[15]


A_s = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10, 10, 11, 11, 11, 12, 12, 12],
        [1, 0.5, 0.5, 1, 0.5, 0.5, 1, 0.5, 0.5], 12, 12)

inv(I-A)

C = rand(12,12)

B = I-A

@benchmark inv(B)

LinearAlgebra.inv!(lu!(B))

#B = 1.0*Matrix(I, 12, 12)

LinearAlgebra.inv!(
    factorize(B))

function tf(A::Array{Float64, 2})
    LinearAlgebra.inv!(factorize(B))
end

@benchmark tf(B)

@benchmark begin
    optimize(
    par -> test.objective(par, test),
    test.par,
    test.optimizer,
    autodiff = :forward,
    Optim.Options())
end

par2 = Optim.minimizer(optimize(
    par -> test.objective(par, test),
    test.par,
    test.optimizer,
    autodiff = :forward,
    Optim.Options(f_tol = 1e-8)
    ))

fit(test)

par = Optim.minimizer(optimize(
    par -> test.objective(par, test),
    test.par,
    test.optimizer,
    autodiff = :forward,
    Optim.Options(iterations = 2)))

@benchmark optimize(
    par -> test.objective(par, test),
    test.par,
    test.optimizer,
    autodiff = :forward,
    Optim.Options(f_tol = 1e-8)
    )

par2 = Optim.minimizer(fit(test))

@code_lowered(test.objective(test.par, test))

par = Feather.read("test/comparisons/three_mean_par.feather")

test.objective(start_values[2], test)


### model 3

test = sem.model(model_funcs[3], datas[3], start_values[3];
    optimizer = LBFGS(; ))

fit(test)

test.objective(test.par, test)

sem.imp_cov(test.ram(start_values[3]))

@benchmark A = [5.0 5.0 5.0 5.0]

convert(ForwardDiff.Dual{Float64}, A)
