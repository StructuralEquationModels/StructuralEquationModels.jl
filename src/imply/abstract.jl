
vars(imply::SemImply) = vars(imply.ram_matrices)
observed_vars(imply::SemImply) = observed_vars(imply.ram_matrices)
latent_vars(imply::SemImply) = latent_vars(imply.ram_matrices)

nvars(imply::SemImply) = nvars(imply.ram_matrices)
nobserved_vars(imply::SemImply) = nobserved_vars(imply.ram_matrices)
nlatent_vars(imply::SemImply) = nlatent_vars(imply.ram_matrices)

params(imply::SemImply) = params(imply.ram_matrices)
nparams(imply::SemImply) = nparams(imply.ram_matrices)

function check_acyclic(A::AbstractMatrix)
    # check if the model is acyclic
    acyclic = isone(det(I - A))

    # check if A is lower or upper triangular
    if istril(A)
        @info "A matrix is lower triangular"
        return LowerTriangular(A)
    elseif istriu(A)
        @info "A matrix is upper triangular"
        return UpperTriangular(A)
    else
        if acyclic
            @warn "Your model is acyclic, specifying the A Matrix as either Upper or Lower Triangular can have great performance benefits.\n" maxlog =
                1
        end
        return A
    end
end
