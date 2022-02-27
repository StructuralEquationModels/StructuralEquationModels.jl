############################################################################
### Types
############################################################################

mutable struct RAM{A1, A2, A3, A4, A5, A6, V, I1, I2, I3, I4, M1, M2, M3, S1, S2, S3, D} <: SemImply
    Σ::A1
    A::A2
    S::A3
    F::A4
    μ::A5
    M::A6

    start_val::V

    A_indices::I1
    S_indices::I2
    M_indices::I3
    I_A_indices::I4

    F⨉I_A⁻¹::M1
    F⨉I_A⁻¹S::M2
    I_A::M3

    ∇A::S1
    ∇S::S2
    ∇M::S3

    identifier::D
end

############################################################################
### Constructors
############################################################################

function RAM(;
        specification,
        start_val = start_fabin3,
        vech = false,
        gradient = true,
        kwargs...)

    # check the model specification
    # if isa(specification, ParameterTable)
    # else if ...
    if specification isa RAMMatrices
        ram_matrices = specification
        identifier = Dict{Symbol, Int64}(ram_matrices.identifier .=> 1:length(ram_matrices.identifier))
    elseif specification isa ParameterTable
        ram_matrices = RAMMatrices(specification)
        identifier = Dict{Symbol, Int64}(ram_matrices.identifier .=> 1:length(ram_matrices.identifier))
    else
        @error "The RAM constructor does not know how to handle your specification object. 
        \n Please specify your model as either a ParameterTable or RAMMatrices."
    end

    A, S, F, M, parameters = 
        ram_matrices.A, ram_matrices.S, ram_matrices.F, ram_matrices.M, ram_matrices.parameters

    if !isa(start_val, Vector)
        start_val = start_val(;ram_matrices = ram_matrices, specification = specification, kwargs...)
    end
    
    n_var, n_nod = size(F)
        
    A = Matrix(A)
    S = Matrix(S)
    F = Matrix(F); F = convert(Matrix{Float64}, F)
    I_A_indices = CartesianIndices(F)

    A_indices = get_parameter_indices(parameters, A)
    S_indices = get_parameter_indices(parameters, S)

    A_indices = [convert(Vector{Int}, indices) for indices in A_indices]
    S_indices = [convert(Vector{Int}, indices) for indices in S_indices]

    A_pre = zeros(size(A)...)
    S_pre = zeros(size(S)...)

    set_constants!(S, S_pre)
    set_constants!(A, A_pre)
    
    # fill copy of a matrix with random parameters
    A_rand = copy(A_pre)
    randpar = rand(length(start_val))

    fill_matrix(
        A_rand,
        A_indices,
        randpar)

    # check if the model is acyclic
    acyclic = isone(det(I-A_rand))

    # check if A is lower or upper triangular
    if iszero(A_rand[.!tril(ones(Bool, size(A)...))])
        A_pre = LowerTriangular(A_pre)
    elseif iszero(A_rand[.!tril(ones(Bool, size(A)...))'])
        A_pre = UpperTriangular(A_pre)
    elseif acyclic
        @info "Your model is acyclic, specifying the A Matrix as either Upper or Lower Triangular can have great performance benefits.\n"
    end

    # pre-allocate some matrices
    Σ = zeros(n_var, n_var)
    F⨉I_A⁻¹ = zeros(n_var, n_nod)
    F⨉I_A⁻¹S = zeros(n_var, n_nod)
    I_A = similar(A_pre)

    if gradient
        ∇A = get_matrix_derivative(A_indices, parameters, n_nod^2)
        ∇S = get_matrix_derivative(S_indices, parameters, n_nod^2)
    else
        ∇A = nothing
        ∇S = nothing
    end

    # μ
    if !isnothing(M)
    
        M_indices = get_parameter_indices(parameters, M)
        M_indices = [convert(Vector{Int}, indices) for indices in M_indices]
    
        M_pre = zeros(size(M)...)
        set_constants!(M, M_pre)
    
        if gradient
            ∇M = get_matrix_derivative(M_indices, parameters, n_nod)
        else
            ∇M = nothing
        end

        μ = zeros(n_var)

    else
        M_indices = nothing
        M_pre = nothing
        μ = nothing
        ∇M = nothing
    end

    return RAM(
        Σ,
        A_pre,
        S_pre,
        F,
        μ,
        M_pre,

        start_val,

        A_indices,
        S_indices,
        M_indices,
        I_A_indices,

        F⨉I_A⁻¹,
        F⨉I_A⁻¹S,
        I_A,

        ∇A,
        ∇S,
        ∇M,

        identifier
    )
end

############################################################################
### functors
############################################################################

function (imply::RAM)(parameters, F, G, H, model)

    fill_A_S_M(
        imply.A, 
        imply.S,
        imply.M,
        imply.A_indices, 
        imply.S_indices,
        imply.M_indices,
        parameters)
    
    imply.I_A .= I - imply.A
    
    if !G
        copyto!(imply.F⨉I_A⁻¹, imply.F)
        rdiv!(imply.F⨉I_A⁻¹, factorize(imply.I_A))
    else
        imply.I_A .= LinearAlgebra.inv!(factorize(imply.I_A))
        imply.F⨉I_A⁻¹ .= imply.F*imply.I_A
    end

    Σ_RAM!(
        imply.Σ,
        imply.F⨉I_A⁻¹,
        imply.S,
        imply.F⨉I_A⁻¹S)

    if !isnothing(imply.μ)
        μ_RAM!(imply.μ, imply.F⨉I_A⁻¹, imply.M)
    end
    
end

############################################################################
### additional functions
############################################################################

function Σ_RAM!(Σ, F⨉I_A⁻¹, S, pre2)
    mul!(pre2, F⨉I_A⁻¹, S)
    mul!(Σ, pre2, F⨉I_A⁻¹')
end

function μ_RAM!(μ, F⨉I_A⁻¹, M)
    mul!(μ, F⨉I_A⁻¹, M)
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::RAM)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end