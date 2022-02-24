############################################################################
### Types
############################################################################

struct RAMSymbolic{F1, F2, F3, A1, A2, A3, S1, S2, S3, V, F4, A4, F5, A5} <: SemImplySymbolic
    Σ_function::F1
    ∇Σ_function::F2
    ∇²Σ_function::F3
    Σ::A1
    ∇Σ::A2
    ∇²Σ::A3
    Σ_symbolic::S1
    ∇Σ_symbolic::S2
    ∇²Σ_symbolic::S3
    start_val::V
    μ_function::F4
    μ::A4
    ∇μ_function::F5
    ∇μ::A5
end

############################################################################
### Constructors
############################################################################

function RAMSymbolic(;
        parameter_table = nothing,
        ram_matrices = nothing,
        loss_types = nothing,
        start_val = start_fabin3,
        vech = false,
        gradient = true,
        hessian = false,
        kwargs...)

    if isnothing(ram_matrices)
        ram_matrices = RAMMatrices(parameter_table)
    end
    
    A, S, F, M, par = 
        ram_matrices.A, ram_matrices.S, ram_matrices.F, ram_matrices.M, ram_matrices.parameters

    if !isa(start_val, Vector)
        start_val = start_val(;ram_matrices = ram_matrices, kwargs...)
    end

    A, S, F = sparse(A), sparse(S), sparse(F)

    if !isnothing(loss_types)
        any(loss_types .<: SemWLS) ? vech = true : nothing
    end

    # Σ
    Σ_symbolic = get_Σ_symbolic_RAM(S, A, F; vech = vech)
    #print(Symbolics.build_function(Σ_symbolic)[2])
    Σ_function = Symbolics.build_function(Σ_symbolic, par, expression=Val{false})[2]
    Σ = zeros(size(Σ_symbolic))
    precompile(Σ_function, (typeof(Σ), Vector{Float64}))

    # ∇Σ
    if gradient
        ∇Σ_symbolic = Symbolics.sparsejacobian(vec(Σ_symbolic), [par...])
        ∇Σ_function = Symbolics.build_function(∇Σ_symbolic, par, expression=Val{false})[2]
        constr = findnz(∇Σ_symbolic)
        ∇Σ = sparse(constr[1], constr[2], fill(1.0, nnz(∇Σ_symbolic)), size(∇Σ_symbolic)...)
        precompile(∇Σ_function, (typeof(∇Σ), Vector{Float64}))
    else
        ∇Σ_symbolic = nothing
        ∇Σ_function = nothing
        ∇Σ = nothing
    end

    if hessian
        n_sig = length(Σ_symbolic)
        n_par = size(par, 1)
        ∇²Σ_symbolic_vec = [Symbolics.sparsehessian(σᵢ, [par...]) for σᵢ in vec(Σ_symbolic)]

        @variables J[1:n_sig]
        ∇²Σ_symbolic = zeros(Num, n_par, n_par)
        for i in 1:n_sig
            ∇²Σ_symbolic += J[i]*∇²Σ_symbolic_vec[i]
        end
    
        ∇²Σ_function = Symbolics.build_function(∇²Σ_symbolic, J, par, expression=Val{false})[2]
        ∇²Σ = zeros(n_par, n_par)
    else
        ∇²Σ_symbolic = nothing
        ∇²Σ_function = nothing
        ∇²Σ = nothing
    end

    # μ
    if !isnothing(M)
        μ_symbolic = get_μ_symbolic_RAM(M, A, F)
        μ_function = Symbolics.build_function(μ_symbolic, par, expression=Val{false})[2]
        μ = zeros(size(μ_symbolic))
        if gradient
            ∇μ_symbolic = Symbolics.jacobian(μ_symbolic, [par...])
            ∇μ_function = Symbolics.build_function(∇μ_symbolic, par, expression=Val{false})[2]
            ∇μ = zeros(size(F, 1), size(par, 1))
        else
            ∇μ_function = nothing
            ∇μ = nothing
        end
    else
        μ_function = nothing
        μ = nothing
        ∇μ_function = nothing
        ∇μ = nothing
    end

    return RAMSymbolic(
        Σ_function,
        ∇Σ_function,
        ∇²Σ_function,
        Σ,
        ∇Σ,
        ∇²Σ,
        Σ_symbolic,
        ∇Σ_symbolic,
        ∇²Σ_symbolic,
        copy(start_val),
        μ_function,
        μ,
        ∇μ_function,
        ∇μ
    )
end

############################################################################
### functors
############################################################################

function (imply::RAMSymbolic)(par, F, G, H, model)
    imply.Σ_function(imply.Σ, par)
    if G || H
        imply.∇Σ_function(imply.∇Σ, par)
    end
    if !isnothing(imply.μ)
        imply.μ_function(imply.μ, par)
        if G || H
            imply.∇μ_function(imply.∇μ, par)
        end
    end
end


############################################################################
### additional functions
############################################################################

function get_Σ_symbolic_RAM(S, A, F; vech = false)
    invia = neumann_series(A)
    Σ_symbolic = F*invia*S*permutedims(invia)*permutedims(F)
    Σ_symbolic = Array(Σ_symbolic)
    # Σ_symbolic = Symbolics.simplify.(Σ_symbolic)
    Threads.@threads for i in eachindex(Σ_symbolic)
        Σ_symbolic[i] = Symbolics.simplify(Σ_symbolic[i])
    end
    if vech Σ_symbolic = Σ_symbolic[tril(trues(size(F, 1), size(F, 1)))] end
    return Σ_symbolic
end

function get_μ_symbolic_RAM(M, A, F)
    invia = neumann_series(A)
    μ_symbolic = F*invia*M
    μ_symbolic = Array(μ_symbolic)
    Threads.@threads for i in eachindex(μ_symbolic)
        μ_symbolic[i] = Symbolics.simplify(μ_symbolic[i])
    end
    return μ_symbolic
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::RAMSymbolic)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end