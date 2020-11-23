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
