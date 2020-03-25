function model(ram, data, par, est = ML, opt = "LBFGS")
    data_matr = convert(Matrix{Float64}, data)

    model = Dict{Symbol, Any}(
        :ram => ram,
        :par => par,
        :data => data_matr,
        :opt => opt,
        :est => est,
        :obs_cov => missing,
        :imp_cov => missing,
        :obs_mean => missing,
        :logl => missing,
        :opt_result => missing
      )
      return model
end
