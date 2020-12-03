function F_missingpattern(imp_mean::Nothing, obs_mean, meandiff,
    pattern, inverse, S, data, ld, mult, n_obs, i)

    # if n_obs == 1
    #     mult = inverse*S
    # else
    #     mul!(mult, inverse, S)
    # end

    mul!(mult[i], inverse[i], S[i])
    F = n_obs[i]*(ld[i] + tr(mult[i]))
    return F
end

function F_missingpattern(imp_mean, obs_mean, meandiff,
    pattern, inverse, S, data, ld, mult, n_obs, i)

    F = n_obs[i]*ld[i]

    @views for j = 1:Int64(n_obs[i])
        @. meandiff[i] = data[i][j, :] - imp_mean[pattern[i]]
        F += meandiff[i]'*inverse[i]*meandiff[i]
    end

    return F
end
