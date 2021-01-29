# minimal working example for fitting a sem in parallel

using LinearAlgebra, Optim, BenchmarkTools

## we need to minimize a loss function that is quite similiar to the one below
# the loss function will hence be called a few hundreds to a few thousend times by an optimizer
# we will often find us in situation where we have several thousend (even millions) of these optimization problems
# but our focus for now is on paralelization of the a single loss function (which is a sum of independend matrix operations)

# number of parallelizable units
# in real life this value will usually be around 500-2000 and almost never > 10000
n = 100 

parameters = [100.0, 300] # we get those from the optimizer; 
                        # it decides which parameters are tested
specific_data = rand(n) # something that is unique between units  

# pre-allocating some memory
# in real life this value will usually be around 10*10 and almost never > 100*100
for i in 1:n pre[i] = pre[i]*pre[i]' end
prod = copy(pre)
a = cholesky.(Hermitian.(pre))

F = zeros(n) # results of the loss function

function loss_perunit(pars, pre, a, specific_data)
    pre[1] = pars[2]*specific_data
    pre[2] = 2*pars[1]*specific_data
    pre[3] = 2*pars[1]*specific_data
    pre[4] = pars[2]*specific_data
    a = cholesky!(Hermitian(pre); check = false)
    if !isposdef(a) return Inf end 
    ld = logdet(a)
    pre = LinearAlgebra.inv!(a)
    return ld + tr(pre)
end

### not parallelized
function loss(pars, pre, a, specific_data, n, F)
    for i in 1:n
        F[i] = loss_perunit(pars, pre[i], a[i], specific_data[i])
    end
    return sum(F)
end

@benchmark loss(parameters, pre, a, specific_data, n, F)

### parallelized
function loss_parallel(pars, pre, a, specific_data, n, F)
    Threads.@threads for i in 1:n
        F[i] = loss_perunit(pars, pre[i], a[i], specific_data[i])
    end
    return sum(F)
end


@benchmark loss_parallel(parameters, pre, a, specific_data, n, F)
