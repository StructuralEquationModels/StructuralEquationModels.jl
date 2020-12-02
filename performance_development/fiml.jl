using sem, Feather, ModelingToolkit, Statistics, LinearAlgebra,
    Optim, SparseArrays, Test, Zygote, LineSearches, ForwardDiff

using Feather, Statistics

abstract type SemObs end

abstract type LossFunction end

miss20_dat = Feather.read("test/comparisons/dat_miss20_dat.feather")
miss30_dat = Feather.read("test/comparisons/dat_miss30_dat.feather")
miss50_dat = Feather.read("test/comparisons/dat_miss50_dat.feather")

miss20_par = Feather.read("test/comparisons/dat_miss20_par.feather")
miss30_par = Feather.read("test/comparisons/dat_miss30_par.feather")
miss50_par = Feather.read("test/comparisons/dat_miss50_par.feather")

miss20_mat = Matrix(miss20_dat)

struct SemObsMissing{
        A <: AbstractArray,
        C <: Union{AbstractArray, Nothing},
        D <: AbstractFloat,
        O <: AbstractFloat,
        P <: Vector,
        R <: Vector,
        PD <: AbstractArray,
        PS <: AbstractArray,
        PO <: AbstractArray} <: SemObs
    data::A
    obs_mean::C
    n_man::D
    n_obs::O
    patterns::P # missing patterns
    rows::R # coresponding rows in the data or matrices
    pattern_data::PD # list of data per missing pattern
    pattern_S::PS
    pattern_n_obs::PO #
end

function SemObsMissing(data; meanstructure = false)

    n_obs = Float64(size(data, 1))
    n_man = Float64(size(data, 2))

    # compute and store the different missing patterns with their rowindices
    missings = ismissing.(data)
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
    n_patterns = size(rows, 1)

    # sort by number of missings
    sort_n_miss = sortperm(sum.(remember))
    remember = remember[sort_n_miss]
    remember_cart = findall.(!, remember)
    rows = rows[sort_n_miss]

    # store the data belonging to the missing patterns
    pattern_data = Vector{Array{Float64}}(undef, n_patterns)
    for i = 1:n_patterns
        pattern_data[i] = miss20_mat[rows[i], remember_cart[i]]
    end
    pattern_data = convert.(Array{Float64}, pattern_data)

    pattern_n_obs = length.(remember_cart)

    # if a meanstructure is needed, don't compute observed means
    if meanstructure
        obs_mean = nothing
        pattern_S = nothing
    else
        pattern_S = Array{Array{Float64, 2}}(undef, length(pattern_data))
        obs_mean = skipmissing_mean(data)

        for i in 1:length(pattern_data)
            S = zeros(pattern_n_obs[i], pattern_n_obs[i])
            for j in 1:size(pattern_data[i], 1)
                diff = pattern_data[i][j, :] - obs_mean[remember_cart[i]]
                S += diff*diff'
            end
            pattern_S[i] = S
        end
    end

    return SemObsMissing(data, obs_mean, n_man, n_obs, remember_cart,
    rows, pattern_data, pattern_S, Float64.(pattern_n_obs))
end

function skipmissing_mean(mat)
    means = Vector{Float64}(undef, size(mat, 2))
    for i = 1:size(mat, 2)
        @views means[i] = mean(skipmissing(mat[:,i]))
    end
    return means
end

@benchmark myobs = SemObsMissing(miss20_mat)

struct SemFIML_Loss{
        INV <: AbstractArray,
        C <: AbstractArray,
        L <: AbstractArray,
        M <: AbstractArray,
        I <: AbstractArray,
        T <: AbstractArray,
        U,
        V} <: LossFunction
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats
    meandiff::M
    imp_inv::I
    mult::T # is this type known?
    objective::U
    grad::V
end

function SemFIML_Loss(observed::O where {O <: SemObs}, objective, grad)

    inverses = broadcast(x -> zeros(x, x), Int64.(observed.pattern_n_obs))
    choleskys = copy(inverses)

    n_patterns = size(observed.rows, 1)
    logdets = zeros(n_patterns)

    meandiff = zeros.(Int64.(observed.pattern_n_obs))

    imp_inv = zeros(Int64(observed.n_obs), Int64(observed.n_obs))
    mult = copy(inverses)

    return SemFIML_Loss(
    inverses,
    choleskys,
    logdets,
    meandiff,
    imp_inv,
    mult,
    copy(objective),
    copy(grad)
    )
end

SemFIML_Loss(myobs, 5.0, 5.0)



## Matrix inversion

LinearAlgebra.lowrankupdate()

## lavaan style matrix updates
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

## structures

# struct SemFIML{O <: SemObs, I <: Imply, L <: Loss, D <: SemDiff} <: AbstractSem
#     observed::O
#     imply::I # former ram
#     loss::L # list of loss functions
#     diff::D
# end
#
# function (model::SemFIML)(par)
#     model.imply(par)
#     F = model.loss(par, model)
#     return F
# end

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
                model.imply.imp_mean,
                model.observed.obs_mean,
                semfiml.meandiff,
                model.observed.patterns,
                semfiml.inverses,
                model.observed.pattern_S,
                model.observed.pattern_data,
                semfiml.logdets,
                semfiml.mult
                model.observed.pattern_n_obs,
                F,
                i
                )
        end
    end
    return F
end

function myf(a)
    x = size(a, 1)
    x = x^2
end

using BenchmarkTools

@benchmark myf([5.0, 6.0])

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

function F_missingpattern(imp_mean::Nothing, obs_mean, meandiff,
    pattern, inverse, S, data, ld, mult, n_obs, F, i)

    # if n_obs == 1
    #     mult = inverse*S
    # else
    #     mul!(mult, inverse, S)
    # end

    mul!(mult, inverse, S)
    F += n_obs*(ld + tr(mult))

end

function F_missingpattern(imp_mean, obs_mean, meandiff,
    pattern, inverse, S, data, ld, mult, n_obs, F, i)

    F += n_obs*logdet

    @views for i = 1:n_obs
        @. meandiff = data[i, :] - imp_mean[pattern]
        F += meandiff'*inverse*meandiff
    end

end

v1 = rand(10)

v2 = rand(10)

ind = [1, 3, 5, 7, 9]

@benchmark @. getindex([v1], ind)

function f1(v1, v2, ind)
    res = v1[ind] - v2[ind]
    return res
end

function f2(v1, v2, ind)
    @views res = v1[ind] - v2[ind]
    return res
end

function f3(v1, v2, ind, res)
    res .= v1[ind] - v2[ind]
    return res
end

function f3(v1, v2, ind, res)
    @inbounds @views @. res = v1[ind] - v2[ind]
    return res
end

function f4(v1, v2, ind, res)
    for i = 1:length(ind)
        res[i] = v1[ind[i]] - v2[ind[i]]
    end
    return res
end

v1_arr = [v1]
v2_arr = [v2]

function f5(v1, v2, ind, res)
    @inbounds res .= view(v1, ind) .- view(v2, ind)
    return res
end

using BenchmarkTools

@benchmark f1(v1, v2, ind)

@benchmark f2(v1, v2, ind)

@benchmark f3($v1, $v2, $ind, $res)

@benchmark f4($v1, $v2, $ind, $res)

@benchmark f5($v1, $v2, $ind, $res)

res = zeros(5)

@code_lowered f3(v1, v2, ind, res)

@code_lowered f4(v1, v2, ind, res)
