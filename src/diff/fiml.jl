struct ∇SemFIML{
        X <: AbstractArray,
        Y <: AbstractArray,
        Z <: AbstractArray,
        W,
        F,
        F2,
        I,
        I2,
        T,
        F3,
        TM,
        F4,
        TBm,
        I3,
        INV <: AbstractArray,
        C <: AbstractArray,
        L <: AbstractArray,
        M <: AbstractArray,
        IM <: AbstractArray,
        I4 <: AbstractArray,
        C2} <: DiffFunction
    B::X
    B!::F
    E::Y
    E!::F2
    F::Z
    C::W
    S_ind_vec::I
    A_ind_vec::I2
    matsize::T
    M!::F3
    M::TM
    Bm!::F4
    Bm::TBm
    M_ind_vec::I3
    inverses::INV #preallocated inverses of imp_cov
    choleskys::C #preallocated choleskys
    logdets::L #logdets of implied covmats
    imp_mean::IM
    meandiff::M
    imp_inv::I4
    counter::C2
end

function ∇SemFIML(
    observed::O where {O <: SemObs}, 
    imply::I where {I <: Imply},
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
        M_ind_vec = nothing
        M_pre = nothing
        Bm_pre = nothing
        fun_mean = nothing
        fun_bm = nothing
    end

    inverses = broadcast(x -> zeros(x, x), Int64.(observed.pattern_nvar_obs))
    choleskys = Array{Cholesky{Float64,Array{Float64,2}},1}(undef, length(inverses))

    n_patterns = size(observed.rows, 1)
    logdets = zeros(n_patterns)

    imp_mean = zeros.(Int64.(observed.pattern_nvar_obs))
    meandiff = zeros.(Int64.(observed.pattern_nvar_obs))
    imp_inv = zeros(size(observed.data, 2), size(observed.data, 2))

    return ∇SemFIML(
        B_pre,
        B!,
        E_pre,
        E!,
        F,
        C_pre,
        S_ind_vec,
        A_ind_vec,
        matsize,
        fun_mean,
        M_pre,
        fun_bm,
        Bm_pre,
        M_ind_vec,
        inverses,
        choleskys,
        logdets,
        imp_mean,
        meandiff,
        imp_inv,
        [0.0])
end

function compute_C(inv_cov, obs_cov, N)
    if N > one(N)
        C = LinearAlgebra.I - inv_cov*obs_cov
        return C
    end
    return reshape([0.0],1,1)
end

function grad_per_pattern(pattern, Σ_inv_j, Σ_der, µ_der, N, C, b)
    Σ_der_j = Σ_der[pattern, pattern]
    µ_der_j = µ_der[pattern]
    if N > one(N)
        d1 = dot(Σ_inv_j,Σ_der_j,C)
    else
        d1 = dot(Σ_inv_j,Σ_der_j)
    end
    d2 = (b'*Σ_inv_j*Σ_der_j + 2*µ_der_j')*Σ_inv_j*b
    d = N*(d1 - d2)
    return d
end

function (diff::∇SemFIML)(par, grad, model::Sem{O, I, L, D}) where
            {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}
    #ld = logdet(a)
    if check_fiml(diff, model)
        copy_per_pattern!(diff, model)
        batch_cholesky!(diff, model)
        diff.logdets .= logdet.(diff.choleskys)
        batch_inv!(diff, model)

        diff.B!(diff.B, par) # B = inv(I-A)
        diff.E!(diff.E, par) # E = B*S*B'
        for i in 1:size(diff.meandiff, 1)
            diff.meandiff[i] .= 
                model.observed.obs_mean[i] - 
                model.imply.imp_mean[model.observed.patterns[i]]
        end

        C = [compute_C(diff.inverses[i], model.observed.obs_cov[i], model.observed.pattern_n_obs[i]) 
                for i in 1:size(diff.meandiff, 1)]

        let B = diff.B, E = diff.E, F = diff.F, Bm = diff.Bm
            Threads.@threads for i = 1:size(par, 1)
                term = similar(diff.C)
                term2 = similar(diff.C)
                sparse_outer_mul!(term, B, E, diff.A_ind_vec[i])
                sparse_outer_mul!(term2, B, B', diff.S_ind_vec[i])
                Σ_der = term2 + term + term'

                term3 = Vector{Float64}(undef, size(B, 1))
                term4 = similar(term3)
                sparse_outer_mul!(term3, B, Bm, diff.A_ind_vec[i])
                sparse_outer_mul!(term4, B, diff.M_ind_vec[i])
                µ_der = term3 + term4

                for j = 1:size(model.observed.patterns, 1)
                    grad[i] += 
                        grad_per_pattern(
                            model.observed.patterns[j], 
                            diff.inverses[j],
                            Σ_der, 
                            µ_der,
                            model.observed.pattern_n_obs[j],
                            C[j],
                            diff.meandiff[j])
                end
            end
        end
    else
        grad .= 0
    end
end