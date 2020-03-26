function model(; ram, data, par, est = ML, opt = "LBFGS", mstruc = false)
    data_matr = convert(Matrix{Float64}, data)

    model = Dict{Symbol, Any}(
        :ram => ram,
        :par => par,
        :data => data_matr,
        :opt => opt,
        :est => est,
        :mstruc => mstruc,
        :obs_cov => missing,
        :imp_cov => missing,
        :obs_mean => missing,
        :logl => missing,
        :opt_result => missing,
        :se => missing,
        :z => missing,
        :p => missing
      )
      return model
end
