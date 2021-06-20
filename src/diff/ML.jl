struct ∇SemML{
        X <: AbstractArray,
        Y <: AbstractArray,
        Z <: AbstractArray,
        W,
        F,
        F2,
        I,
        I2,
        T} <: DiffFunction
    B::X
    B!::F
    E::Y
    E!::F2
    F::Z
    C::W
    S_ind_vec::I
    A_ind_vec::I2
    matsize::T
end

function ∇SemML(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val;
        M = nothing
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC
            }

    A = copy(A)
    S = copy(S)
    F = copy(F)


    invia = neumann_series(A)

    #imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
    #imp_cov_sym = Array(imp_cov_sym)
    invia = ModelingToolkit.simplify.(invia)
    B = invia
    E = B*S*B'
    E = E*permutedims(F)
    E = Array(E)
    E = ModelingToolkit.simplify.(E)
    B = F*B
    B = Array(B)
    B = ModelingToolkit.simplify.(B)

    B! =
        eval(ModelingToolkit.build_function(
            B,
            parameters
            )[2])

    E! =
        eval(ModelingToolkit.build_function(
            E,
            parameters
            )[2])

    B_pre = zeros(size(F)...)
    E_pre = zeros(size(F, 2), size(F, 1))

    grad = similar(start_val)
    matsize = size(A)
    C_pre = zeros(size(F, 1), size(F, 1))

    #S_ind_vec = Vector{Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}}()
    #A_ind_vec = Vector{Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}}()
    S_ind_vec = Vector{Vector{CartesianIndex{2}}}()
    A_ind_vec = Vector{Vector{CartesianIndex{2}}}()

    for i = 1:size(parameters, 1)
        S_ind = findall(var -> isequal(var, parameters[i]), S)
        A_ind = findall(var -> isequal(var, parameters[i]), A)

        push!(S_ind_vec, S_ind)
        push!(A_ind_vec, A_ind)
    end


    if !isnothing(M)
        
        fun_mean =
            eval(ModelingToolkit.build_function(
                M,
                parameters
            )[2])
        M_pre = zeros(size(F)[2])

        Bm = B*M
        Bm = ModelingToolkit.simplify.(Bm)
        fun_bm =
            eval(ModelingToolkit.build_function(
                Bm,
                parameters
            )[2])
        Bm_pre = zeros(size(F)[2])

        M_ind_vec = Vector{Vector{Int64}}()
        for i = 1:size(parameters, 1)
            M_ind = findall(var -> isequal(var, parameters[i]), M)
            push!(M_ind_vec, M_ind)
        end
    else
        imp_fun_mean = nothing
        imp_mean = nothing
    end

    return ∇SemML(
        B_pre,
        B!,
        E_pre,
        E!,
        F,
        C_pre,
        S_ind_vec,
        A_ind_vec,
        matsize)
end

function (diff::∇SemML)(par, grad, model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}
    a = cholesky(Hermitian(model.imply.imp_cov); check = false)
    if !isposdef(a)
        grad .= 0
    else
        #ld = logdet(a)
        Σ_inv = inv(a)
        diff.B!(diff.B, par) # B = inv(I-A)
        diff.E!(diff.E, par) # E = B*S*B'
        let B = diff.B, E = diff.E, F = diff.F, D = model.observed.obs_cov
            #C = LinearAlgebra.I-Σ_inv*D
            mul!(diff.C, Σ_inv, D)
            diff.C .= LinearAlgebra.I-diff.C
            Threads.@threads for i = 1:size(par, 1)
                term = similar(diff.C)
                term2 = similar(diff.C)
                sparse_outer_mul!(term, B, E, diff.A_ind_vec[i])
                sparse_outer_mul!(term2, B, B', diff.S_ind_vec[i])
                Σ_der = term2 + term + term'
                grad[i] = tr(Σ_inv*Σ_der*diff.C)
            end
        end
    end
end