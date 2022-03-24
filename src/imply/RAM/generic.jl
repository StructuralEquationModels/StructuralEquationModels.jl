############################################################################
### Types
############################################################################

mutable struct RAM{A1, A2, A3, A4, A5, A6, V, V2, I1, I2, I3, M1, M2, M3, S1, S2, S3, D} <: SemImply
    Σ::A1
    A::A2
    S::A3
    F::A4
    μ::A5
    M::A6

    n_par::V
    ram_matrices::V2

    A_indices::I1
    S_indices::I2
    M_indices::I3

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
        vech = false,
        gradient = true,
        kwargs...)

    # check the model specification
    # if isa(specification, ParameterTable)
    # else if ...
    if specification isa RAMMatrices
        ram_matrices = specification
        identifier = StructuralEquationModels.identifier(ram_matrices)
    elseif specification isa ParameterTable
        ram_matrices = RAMMatrices(specification)
        identifier = StructuralEquationModels.identifier(ram_matrices)
    else
        @error "The RAM constructor does not know how to handle your specification object. 
        \n Please specify your model as either a ParameterTable or RAMMatrices."
    end

    # get dimensions of the model
    n_par = length(ram_matrices.parameters)
    n_var, n_nod = ram_matrices.size_F
    parameters = ram_matrices.parameters
    F = zeros(ram_matrices.size_F); F[CartesianIndex.(1:n_var, ram_matrices.F_ind)] .= 1.0

    # get indices
    A_indices = copy(ram_matrices.A_ind)
    S_indices = copy(ram_matrices.S_ind)
    !isnothing(ram_matrices.M_ind) ? M_indices = copy(ram_matrices.M_ind) : M_indices = nothing

    #preallocate arrays
    A_pre = zeros(n_nod, n_nod)
    S_pre = zeros(n_nod, n_nod)
    !isnothing(M_indices) ? M_pre = zeros(n_nod) : M_pre = nothing

    set_RAMConstants!(A_pre, S_pre, M_pre, ram_matrices.constants)
    
    A_pre = check_acyclic(A_pre, n_par, A_indices)

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
    if !isnothing(M_indices)

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

        n_par,
        ram_matrices,

        A_indices,
        S_indices,
        M_indices,

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
### Recommended methods
############################################################################

identifier(imply::RAM) = imply.identifier
n_par(imply::RAM) = imply.n_par

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

function check_acyclic(A_pre, n_par, A_indices)
    # fill copy of A-matrix with random parameters
    A_rand = copy(A_pre)
    randpar = rand(n_par)

    fill_matrix(
        A_rand,
        A_indices,
        randpar)

    # check if the model is acyclic
    acyclic = isone(det(I-A_rand))

    # check if A is lower or upper triangular
    if iszero(A_rand[.!tril(ones(Bool, size(A_pre)...))])
        A_pre = LowerTriangular(A_pre)
    elseif iszero(A_rand[.!tril(ones(Bool, size(A_pre)...))'])
        A_pre = UpperTriangular(A_pre)
    elseif acyclic
        @info "Your model is acyclic, specifying the A Matrix as either Upper or Lower Triangular can have great performance benefits.\n"
    end

    return A_pre
end

############################################################################
### Pretty Printing
############################################################################

function Base.show(io::IO, struct_inst::RAM)
    print_type_name(io, struct_inst)
    print_field_types(io, struct_inst)
end