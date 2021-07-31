function neumann_series(mat::SparseMatrixCSC)
    inverse = I + mat
    next_term = mat^2

    while nnz(next_term) != 0
        inverse += next_term
        next_term *= mat
    end

    return inverse
end

function make_onelement_array(A)
    isa(A, Array) ? nothing : (A = [A])
    return A
end

function semvec(observed, imply, loss, diff)

    observed = make_onelement_array(observed)
    imply = make_onelement_array(imply)
    loss = make_onelement_array(loss)
    diff = make_onelement_array(diff)

    #sem_vec = Array{AbstractSem}(undef, maximum(length.([observed, imply, loss, diff])))
    sem_vec = Sem.(observed, imply, loss, diff)

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

function skipmissing_mean(mat)
    means = Vector{Float64}(undef, size(mat, 2))
    for i = 1:size(mat, 2)
        @views means[i] = mean(skipmissing(mat[:,i]))
    end
    return means
end

function F_one_person(imp_mean, meandiff, inverse, data, logdet)
    F = logdet
    @. meandiff = data - imp_mean
    F += dot(meandiff, inverse, meandiff)
    return F
end

function remove_all_missing(data)
    keep = Vector{Int64}()
    for i = 1:size(data, 1)
        if any(.!ismissing.(data[i, :]))
            push!(keep, i)
        end
    end
    return data[keep, :], keep
end

function batch_inv!(fun::Union{LossFunction, DiffFunction}, model)
    for i = 1:size(fun.inverses, 1)
        fun.inverses[i] .= LinearAlgebra.inv!(fun.choleskys[i])
    end
end

function sparse_outer_mul!(C, A, B, ind) #computes A*S*B -> C, where ind gives the entries of S that are 1
    fill!(C, 0.0)
    for i in 1:length(ind)
        BLAS.ger!(1.0, A[:, ind[i][1]], B[ind[i][2], :], C)
    end
end

function sparse_outer_mul!(C, A, ind) #computes A*∇m, where ∇m ind gives the entries of ∇m that are 1
    fill!(C, 0.0)
    @views C .= sum(A[:, ind], dims = 2)
    return C
end

function sparse_outer_mul!(C, A, B::Vector, ind) #computes A*S*B -> C, where ind gives the entries of S that are 1
    fill!(C, 0.0)
    @views @inbounds for i in 1:length(ind)
        C .+= B[ind[i][2]].*A[:, ind[i][1]]
    end
end

function cov_and_mean(rows; corrected = false)
    data = permutedims(hcat(rows...))
    size(rows, 1) > 1 ?
        obs_cov = Statistics.cov(data; corrected = corrected) :
        obs_cov = reshape([0.0],1,1)
    obs_mean = vcat(Statistics.mean(data, dims = 1)...)
    return obs_cov, obs_mean
end