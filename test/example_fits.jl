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


test = sem.model(ram(three_mean_func(start_values)[1],
                        three_mean_func(start_values)[2],
                        three_mean_func(start_values)[3],),
                ramfunc,        
                datas[2],
                start_values[2])

test.ram(start_values[2])

start_values[2]

ms = (test.ram(test.par))

@benchmark sem.imp_cov(test.ram(test.par))

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

A = [0  0  0  0  0  0  0  0  0  1     0     0.0
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

@benchmark A[1,2] = 5

function tf(A::Array{Float64, 2})
    inv(I-A)
end

@benchmark tf(A)

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
