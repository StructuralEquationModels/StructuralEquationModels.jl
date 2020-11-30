using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff

miss20_dat = Feather.read("test/comparisons/dat_miss20_dat.feather")
miss30_dat = Feather.read("test/comparisons/dat_miss30_dat.feather")
miss50_dat = Feather.read("test/comparisons/dat_miss50_dat.feather")

miss20_par = Feather.read("test/comparisons/dat_miss20_par.feather")
miss30_par = Feather.read("test/comparisons/dat_miss30_par.feather")
miss50_par = Feather.read("test/comparisons/dat_miss50_par.feather")

miss20_mat = Matrix(miss20_dat)

# missings = findall(ismissing, miss20_mat)
#
# #indic = Vector{Array}(undef, size(miss20_mat, 1))
#
# missings[sortperm(getindex.(missings, 1))]
#
# #rows = getindex.(missings, 1)
# #cols = getindex.(missings, 2)
#
# ind_3 = findall(x -> x[1] == 3, missings)
# rel_miss = missings[ind_3]
# getindex.(rel_miss, 2)
#
# ind_per_row = Vector{Vector{Int64}}(undef, size(miss20_mat, 1))
#
# for i = 1:size(miss20_mat, 1)
#     ind_i = findall(x -> x[1] == i, missings)
#     rel_miss = missings[ind_i]
#     ind_per_row[i] = getindex.(rel_miss, 2)
# end
#
# rows = sortperm(length.(ind_per_row))
#
# ind_per_row = ind_per_row[sortperm(length.(ind_per_row))]
#
# ind_i = findall(x -> length(x)==2, ind_per_row)
# rel_ind = ind_per_row[ind_i]
# size(rel_ind, 1)
# rel_ind
# sortperm(permutedims(hcat(rel_ind...)), dims = 1)
#
# for i = 0:size(miss20_mat, 1)
#     ind_i = findall(x -> length(x)==i, ind_per_row)
#     rel_ind = ind_per_row[ind_i]
#     for i = 1:size(rel_ind, 1)
# end

missings = ismissing.(miss20_mat)

patterns = [missings[i, :] for i = 1:size(missings, 1)]
remember = Vector{BitArray{1}}()
rows = [Vector{Int64}(undef, 0) for i = 1:size(patterns, 1)]


for i = 1:size(patterns, 1)
    unknown = true
    for j = 1:size(remember, 1)
        if patterns[i] == remember[j]
            push!(rows[j], i)
            unknown = false
        end
    end
    if unknown
        push!(remember, patterns[i])
        push!(rows[size(remember, 1)], i)
    end
end

rows = rows[1:length(remember)]

remember[2]

rows

sort_n_miss = sortperm(sum.(remember))

remember = remember[sort_n_miss]

remember_cart = findall.(!, remember)

rows = rows[sort_n_miss]

n_miss = sum.(remember)[sort_n_miss]

covmat = rand(11,11); covmat = covmat*covmat'

function invert_submatrix(mat, cache, ind)
    cache .= @view mat[ind, ind]
    a = cholesky!(cache)

end

cache = rand(10,10)

@benchmark invert_submatrix($covmat, $cache, $remember_cart[2])

remember

## Matrix inversion

LinearAlgebra.lowrankupdate()

##
S = rand(20,20)

S = S*transpose(S)

save = copy(S)

S_inv = inv(S)

rm_idx = [3,4]
not_idx = vcat(1,2,collect(5:20))

function lv_inv(S_inv, rm_idx, not_idx)
    A = S_inv[rm_idx, not_idx]
    H = S_inv[rm_idx,  rm_idx]
    out = S_inv[not_idx, not_idx] - transpose(A)*(H\A)
    return out
end

function my_inv(S, not_idx)
    myinv = inv(cholesky(S[not_idx, not_idx]))
    return myinv
end

@benchmark lv_inv(S_inv, rm_idx, not_idx)
@benchmark my_inv(S, not_idx)

S_del = copy(S)
S_del[1, :] .= 0.0
S_del[:, 1] .= 0.0
S_del[1,1] = 1

inv(S_del)[2:20, 2:20] ≈ inv(S[2:20, 2:20])

out

inv(S[not_idx, not_idx])

[missing, 5.0, 5.0]

mat1 = [rand(10,10) for i = 1:10]
mat2 = [zeros(12,12) for i = 1:10]
ind = [1,2,3,4,5,6,8,9,10,11]

copyto!(mat1, mat2[ind, ind])

@benchmark copyto!(mat1, mat2[ind, ind])

function myview(mat1, mat2, ind)
    @views for i = 1:10
        mat1[i] .= mat2[i][ind, ind]
    end
    return mat1
end

myview(mat1, mat2, ind)

@benchmark myview($mat1, $mat2, $ind)

covmat = rand(10,10); covmat = covmat*covmat'

covmat2 = rand(10,10); covmat2 = covmat2*covmat2'

vec1 = [covmat, covmat2]

vec2 = [covmat, covmat2]


@benchmark copyto!(vec1, vec2)

a = cholesky!(covmat)
b = cholesky!(covmat2)

[a,b]

dest = Vector{Cholesky{Float64,Array{Float64,2}}}(undef, 2)

mychol(dest, covmat)

logdet_vec = zeros(2)

function logdets(vec, dest)
    vec .= logdet.(dest)
end

function logdets2(vec, dest)
    for i = 1:2
        vec[i] = logdet(dest[i])
    end
end

@benchmark logdets(logdet_vec, dest)

@benchmark logdets2(logdet_vec, dest)

a = rand(20, 10)

a = Array{Float64, 2}(undef, 1, 1)

a[1] = 0.5
cov(a)

mean(a, dims = 1)

skipmissing(a)


@benchmark mapslices(mean, $a, dims = 1)

function skipmissing_mean(mat)
    means = Vector{Float64}(undef, size(mat, 2))
    for i = 1:size(mat, 2)
        @views means[i] = mean(skipmissing(mat[:,i]))
    end
    return means
end

@benchmark skipmissing_mean(a)


## structures

struct SemFIML{O <: SemObs, I <: Imply, L <: Loss, D <: SemDiff} <: AbstractSem
    observed::O
    imply::I # former ram
    loss::L # list of loss functions
    diff::D
end

function (model::SemFIML)(par)
    model.imply(par)
    F = model.loss(par, model)
    return F
end

struct SemFIML_Loss{I <: AbstractArray, T <: AbstractArray, U, V} <: LossFunction
    patterns #missing patterns
    rows #coresponding rows in the data or matrices
    inverses #preallocated inverses of imp_cov
    choleskys #preallocated choleskys
    logdets #logdets of implied covmats
    obs_mean #observed means
    imp_inv::I
    mult::T # is this type known?
    objective::U
    grad::V
end

function SemFIML_Loss(observed::T, objective, grad) where {T <: SemObs}
    return SemFIML_Loss(
        copy(observed.obs_cov),
        copy(observed.obs_cov),
        copy(objective),
        copy(grad)
        )
end


function (semfiml::SemFIML_Loss)(par, model::Sem{O, I, L, D}) where
            {O <: SemObs, L <: Loss, I <: Imply, D <: SemFiniteDiff}


    copyto!(semfiml.inverses[1], model.imply.imp_cov)
    semfiml.choleskys[1] = cholesky!(Hermitian(semfiml.inverses[1]); check = false)

    if !isposdef(semfiml.choleskys[1])
        F = Inf
    else
        @views for i = 2:size(inverses, 1)
            semfiml.inverses[i] .= model.imply.imp_cov[patterns[i], patterns[i]]
        end

        for i = 2:size(inverses, 1)
            semfilm.choleskys[i] = cholesky!(semfiml.inverses[i]; check = false)
        end

        #ld = logdet(a)
        logdets .= logdet.(semfiml.choleskys)

        #semml.imp_inv .= LinearAlgebra.inv!(a)
        for i = 1:size(inverses, 1)
            semfiml.inverses[i] .= LinearAlgebra.inv!(semfiml.choleskys[i])
        end

        F = zero(eltype(par))

        for i = 1:size(semfiml.rows, 1)
            F_missingpattern(
                    #semfiml.rows[i],
                    semfiml.patterns[i],
                    semfiml.inverses[i],
                    semfiml.observed[i],
                    semfiml.logdets[i],
                    semfiml.mult[i]
                    convert(Float64, size(rows, 1)),
                    F
                    #model.imply.imp_mean
                    )
        end

    end
    return F
end

# function F_missingpattern(rows, pattern, inverse, S, logdet, n_obs)
#     if size(rows, 1) == n_obs # no missings
#         F =
#     else if size(rows, 1) > 5 # find a good treshold
#
#     else if size(rows, 1) == 1 #no one else has the specific missing pattern
#
#     else #few persons, so better compute individually
#
#     end
#     return F
# end

function F_missingpattern(pattern, inverse, S, ld, mult, n_obs)

    if n_obs == 1
        mult = inverse*S
    else
        mul!(mult, inverse, S)
    end

    F += n_obs*ld + tr(mult)

end

function F_missingpattern(pattern, inverse, data, ld, mult, n_obs, imp_mean)
    F += n_obs*logdet
    @views for i = 1:n_obs
        F +=
            (data[i, :] - imp_mean[pattern])'*
            inverse*
            (data[i, :] - imp_mean[pattern])
    end
end


using LinearAlgebra

?LinearAlgebra.BLAS.spmv!

A = rand(5)

B = rand(5,5)

pre = zeros(5)

mul!(pre, transpose(A),B)

function myf(A, B)
    res = LinearAlgebra.BLAS.gemv('N', B, A)
    A'*res
end


function myf2(A, B)
    transpose(A)*B*A
end

@code_native myf(A, B)

@code_llvm myf2(A, B)

@benchmark myf(A, B)

@benchmark myf2(A, B)

res = LinearAlgebra.BLAS.gemv('N', B, A)

LinearAlgebra.BLAS.gemv('T', A, res)

A'*res


struct mystr2{A <: Any}
    f::A
end

test = mystr2(10)

test2 = mystr2(nothing)

function (str::mystr2{B} where {B <: Nothing})()
end
function (str::mystr2)()
    return 5.0
end

test()
