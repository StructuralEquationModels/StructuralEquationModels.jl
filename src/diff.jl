struct SemForwardDiff <: SemDiff
    algorithm
    options #Optim.Options() call to optimize()
end

# function SemForwardDiff()
#     return SemForwardDiff(Optim.Options())
# end

struct SemReverseDiff{} <: SemDiff
    algorithm
    options
    # For preallocations, see the examples in ReverseDiff
end

struct SemFiniteDiff{} <: SemDiff
    algorithm
    options
end

# function SemFiniteDiff()
#     return SemFiniteDiff(Optim.Options())
# end

struct SemAnalyticDiff{
    A <: AbstractArray,
    X <: SparseMatrixCSC,
    Y <: SparseMatrixCSC,
    Z <: SparseMatrixCSC,
    T} <: SemDiff
    algorithm
    options
    grad::A
    B::X
    B!
    E::Y
    E!
    F::Z
    S_ind_vec
    A_ind_vec
    matsize::T
end

function SemAnalyticDiff(
        algorithm,
        options,
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC
            }

    invia = I + A
    next_term = A^2

    while nnz(next_term) != 0
        invia += next_term
        next_term *= next_term
    end

    #imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
    #imp_cov_sym = Array(imp_cov_sym)
    invia .= ModelingToolkit.simplify.(invia)
    B = invia
    E = B*S*B'
    E .= ModelingToolkit.simplify.(E)

    B_, B! =
        eval.(ModelingToolkit.build_function(
            B,
            parameters
            ))

    E_, E! =
        eval.(ModelingToolkit.build_function(
            E,
            parameters
            ))

    B_pre = Base.invokelatest(B_, start_val)
    E_pre = Base.invokelatest(E_, start_val)
    imp_cov = rand(size(F)[1], size(F)[1])

    grad = similar(start_val)
    matsize = size(A)

    for i = 1:length(A)
        if !isa(A[i], Operation)
            A[i] = ModelingToolkit.Constant(0)
        end
    end
    SparseArrays.dropzeros!(A)

    for i = 1:length(S)
        if !isa(S[i], Operation)
            S[i] = ModelingToolkit.Constant(0)
        end
    end

    SparseArrays.dropzeros!(S)


    S = Array(S)
    A = Array(A)

    S_dense! =
        eval(ModelingToolkit.build_function(
            S,
            parameters
            )[2])

    A_dense! =
        eval(ModelingToolkit.build_function(
            A,
            parameters
            )[2])

    S_ind_vec = Vector{Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}}()
    A_ind_vec = Vector{Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}}()

    d = zeros(size(start_val, 1))

    for i = 1:size(start_val, 1)
        d .= 0.0
        d[i] = 1.0

        S_der = zeros(matsize)
        A_der = zeros(matsize)

        Base.invokelatest(S_dense!, S_der, d)
        Base.invokelatest(A_dense!, A_der, d)

        S_der = sparse(S_der)
        A_der = sparse(A_der)

        S_ind = findnz(S_der)
        A_ind = findnz(A_der)

        push!(S_ind_vec, S_ind)
        push!(A_ind_vec, A_ind)
    end

    return SemAnalyticDiff(
        algorithm,
        options,
        grad,
        B_pre,
        B!,
        E_pre,
        E!,
        F,
        S_ind_vec,
        A_ind_vec,
        matsize)
end
