using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff

## Observed Data
three_path_dat = Feather.read("test/comparisons/three_path_dat.feather")
three_path_par = Feather.read("test/comparisons/three_path_par.feather")
three_path_start = Feather.read("test/comparisons/three_path_start.feather")

semobserved = SemObsCommon(data = Matrix(three_path_dat))

diff_fin = SemFiniteDiff(BFGS(), Optim.Options())


## Model definition
@variables x[1:31]

S =[x[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
    0     x[2]  0     0     0     0     0     0     0     0     0     0     0     0
    0     0     x[3]  0     0     0     0     0     0     0     0     0     0     0
    0     0     0     x[4]  0     0     0     x[15] 0     0     0     0     0     0
    0     0     0     0     x[5]  0     x[16] 0     x[17] 0     0     0     0     0
    0     0     0     0     0     x[6]  0     0     0     x[18] 0     0     0     0
    0     0     0     0     x[16] 0     x[7]  0     0     0     x[19] 0     0     0
    0     0     0     x[15] 0     0     0     x[8]  0     0     0     0     0     0
    0     0     0     0     x[17] 0     0     0     x[9]  0     x[20] 0     0     0
    0     0     0     0     0     x[18] 0     0     0     x[10] 0     0     0     0
    0     0     0     0     0     0     x[19] 0     x[20] 0     x[11] 0     0     0
    0     0     0     0     0     0     0     0     0     0     0     x[12] 0     0
    0     0     0     0     0     0     0     0     0     0     0     0     x[13] 0
    0     0     0     0     0     0     0     0     0     0     0     0     0     x[14]]

F =[1.0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 0 0]

A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[21] 0     0
    0  0  0  0  0  0  0  0  0  0  0     x[22] 0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0  0  0     0     x[23] 0
    0  0  0  0  0  0  0  0  0  0  0     0     x[24] 0
    0  0  0  0  0  0  0  0  0  0  0     0     x[25] 0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[26]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[27]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[28]
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[29] 0     0
    0  0  0  0  0  0  0  0  0  0  0     x[30] x[31] 0]


S = sparse(S)

#F
F = sparse(F)

#A
A = sparse(A)

start_val = vcat(
    vec(var(Matrix(three_path_dat), dims = 1))./2,
    fill(0.05, 3),
    fill(0.0, 6),
    fill(1.0, 8),
    fill(0, 3)
    )

loss = Loss([SemML(semobserved, [0.0], similar(start_val))])

imply = ImplySymbolic(A, S, F, x, start_val)

model_fin = Sem(semobserved, imply, loss, diff_fin)

A = Vector{Any}(undef, 6)
for i in 1:6
    global A[i] = deepcopy(imply)
end

##


rowind = [1:25, collect(26:50), collect(51:75)]

differ_group = [
    fill(1, 31),
    vcat(fill(1, 14), fill(2, 17)),
    vcat(fill(1, 7), fill(2, 7), fill(3, 10), fill(2, 7))
    ]

function parsubset(differ_group, start_val)
    differ_group = hcat(differ_group...)
    npar, ngroup = size(differ_group)

    parunique = Vector{Int64}(undef, npar)
    for i in 1:npar
        parunique[i] = length(unique(differ_group[i, :]))
    end

    npar_effective = sum(parunique)

    locations = zeros(Int64, ngroup, npar)
    for i in 1:ngroup
        locations[i, :] .=
            cumsum(parunique) .- parunique .+ differ_group[:, i]
    end

    start_val_long = vcat(fill.(start_val, parunique)...)

    parsubsets = zeros(Bool, ngroup, npar_effective)
    for i in 1:ngroup
        parsubsets[i, locations[i, :]] .= true
    end

    return parsubsets, start_val_long
end

parsubset(differ_group, start_val)

function make_onelement_array(A)
    isa(A, Array) ? nothing : (A = [A])
    return A
end

function semvec(observed, imply, loss, diff)

    observed = make_onelement_array(observed)
    imply = make_onelement_array(imply)
    loss = make_onelement_array(loss)
    diff = make_onelement_array(diff)

    sem_vec = Array{AbstractSem}(undef, maximum(length.([observed, imply, loss, diff])))
    sem_vec .= Sem.(observed, imply, loss, diff)

    return sem_vec
end

function get_observed(rowind, data, semobserved;
            args = (),
            kwargs = NamedTuple())
    observed_vec = Vector{semobserved}(undef, length(rowind))
    for i in 1:length(rowind)
        observed_vec[i] = semobserved(
                            args...;
                            data = Matrix(data[rowind[i], :]),
                            kwargs...)
    end
    return observed_vec
end


function start_val(multigroup_sem, start_val)
end

function SemMG(
    data, start_val, differ_group, rowind,
    observed, imply, loss, diff;
    obs_args = (),
    imply_agrs = (),
    loss_args = (),
    diff_args = ())

    #diff = diff(diff_args...)
    #imply = imply(imply_args...)
    #loss = loss(loss_args...)
    #observed = observed(obs_args...)

    sem_vec = semvec(observed, imply, loss, diff)

    par_subsets, start_val_long = parsubset(differ_group, start_val)


    return SemMG2(sem_vec, par_subsets), start_val_long
end


obs_list = get_observed(rowind, three_path_dat, SemObsCommon)

sem_mg, start_val_mg = SemMG(three_path_dat, start_val, differ_group, rowind, obs_list,
    imply, loss, diff_fin)

sem_mg

function (semmg::SemMG)(par)
    F = zero(eltype(par))
    for i in 1:length(semmg.sem_vec)
        F += semmg.sem_vec[i](par[sem.par_subsets[i,:]])
    end
    return F
end

start_val_mg[sem_mg.par_subsets[3,:]] == start_val


## Testing Zone

function MultigroupSem(
    data, algorithm, options, A, S, F, x, start_val,
    differ_group, rowind,
    semobserved, imply, loss, diff;
    obs_args = NamedTuple(),
    imply_agrs = NamedTuple(),
    loss_args = NamedTuple(),
    diff_args = NamedTuple())

    diff = diff(algorithm, options, )
end

function fitfun(par, models, parsubsets)
    F = zero(eltype(par))
    for i in 1:length(models)
        F += models[i](par[parsubsets[i,:]])
    end
    return F
end


function test(diff, algo; args1 = NamedTuple())
    return diff(;algo, args1...)
end

function test2(;algo = nothing, data = nothing)
    return data
end

test(test2, "a")

Dict([("data", 4)])

x = (data = 10,)

x

function test3(;data)
    return data
end


test3(;x...)

maximum(length.([1 2 3]))

length(semobserved)


function f(constructor, value, length)
    vec = Vector{constructor}(undef, length)
    for i = 1:length
        vec[i] = constructor(value)
    end
end


struct SemMG2{V <: Tuple{AbstractSem}, D <: Array{Bool, 2}} <: AbstractSem
    sem_vec::V
    par_subsets::D
end
