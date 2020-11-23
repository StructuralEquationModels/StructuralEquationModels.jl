function parsubset(differ_group, start_val)
    differ_group = hcat(differ_group...)
    npar, ngroup = size(differ_group)

    parunique = Vector{Int64}(undef, npar)
    for i in 1:npar
        parunique[i] = length(unique(differ_group[i, :]))
    end

    npar_effective = sum(parunique)

    locations = zeros(Int64, ngroup, npar)
    for i in 1:ngroup
        locations[i, :] .=
            cumsum(parunique) .- parunique .+ differ_group[:, i]
    end

    locations = [locations[i, :] for i in 1:size(locations, 1)]

    start_val_long = vcat(fill.(start_val, parunique)...)

    # parsubsets = falses(ngroup, npar_effective)
    # parsubsets = [parsubsets[i, :] for i in 1:size(parsubsets, 1)]

    # for i in 1:ngroup
    #     parsubsets[i][locations[i, :]] .= true
    # end

    return locations, start_val_long
end

function MGSem(
    data, start_val, differ_group, rowind,
    observed, imply, loss, diff;
    obs_args = (),
    imply_agrs = (),
    loss_args = (),
    diff_args = ())

    #diff = diff(diff_args...)
    #imply = imply(imply_args...)
    #loss = loss(loss_args...)
    #observed = observed(obs_args...)

    sem_vec = semvec(observed, imply, loss, diff)

    par_subsets, start_val_long = parsubset(differ_group, start_val)


    return MGSem(sem_vec, par_subsets), start_val_long
end

function (semmg::MGSem)(par)
    F = zero(eltype(par))
    for i in 1:length(semmg.sem_vec)
        F += (semmg.sem_vec[i].observed.n_obs-1)*
            semmg.sem_vec[i](view(par, semmg.par_subsets[i]))
    end
    return F
end
