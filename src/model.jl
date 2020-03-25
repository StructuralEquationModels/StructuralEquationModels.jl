function model(ram, data, parameters, est = ML, optim = "LBFGS")
    data_matr = convert(Matrix{Float64}, data)
    obs_cov = cov(data_matr)
    obs_means::Vector{Float64} = vec(mean(data_matr, dims = 1))

    model = Dict{Symbol, Any}(
      :parameters => parameters,
      :data => data_matr,
      :obs_cov => obs_cov,
      :exp_cov => expected_cov(model, parameters),
      :obs_means => obs_means,
      :ram => ram,
      :logl => logl(obs_means, exp_cov, data_matr),
      :opt_result => result,
      :optimizer => optim,
      :start => start
      )
      return model
end
