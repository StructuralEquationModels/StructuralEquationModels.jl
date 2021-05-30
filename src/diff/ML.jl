struct ∇SemML{
        X <: AbstractArray,
        Y <: AbstractArray,
        Z <: AbstractArray,
        T} <: DiffFunction
    B::X
    B!
    E::Y
    E!
    F::Z
    S_ind_vec
    A_ind_vec
    matsize::T
end

function ∇SemML(
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

    A = copy(A)
    S = copy(S)
    F = copy(F)


    invia = neumann_series(A)

    #imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
    #imp_cov_sym = Array(imp_cov_sym)
    invia = ModelingToolkit.simplify.(invia)
    B = invia
    E = B*S*B'
    E = Array(E)
    E = ModelingToolkit.simplify.(E)
    B = Array(B)

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

    B_pre = zeros(size(S)...)
    E_pre = zeros(size(S)...)

    grad = similar(start_val)
    matsize = size(A)

    S_ind_vec = Vector{Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}}()
    A_ind_vec = Vector{Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}}()

    for i = 1:size(parameters, 1)
        S_der = broadcast(var -> Float64(isequal(var, parameters[i])), S)
        A_der = broadcast(var -> Float64(isequal(var, parameters[i])), A)

        S_der = sparse(S_der)
        A_der = sparse(A_der)

        S_ind = findnz(S_der)
        A_ind = findnz(A_der)

        push!(S_ind_vec, S_ind)
        push!(A_ind_vec, A_ind)
    end

    return ∇SemML(
        B_pre,
        B!,
        E_pre,
        E!,
        F,
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
            for i = 1:size(par, 1)
                S_der = sparse(diff.S_ind_vec[i]..., diff.matsize...)
                A_der = sparse(diff.A_ind_vec[i]..., diff.matsize...)

                term = F*B*A_der*E*F'
                Σ_der = Array(F*B*S_der*B'F' + term + term')

                grad[i] = tr(Σ_inv*Σ_der) + tr((-Σ_inv)*Σ_der*Σ_inv*D)
            end
        end
    end
end