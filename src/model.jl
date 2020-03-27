function model(; ram, data, par,
                    est = ML,
                    opt = "LBFGS",
                    mstruc = false,
                    reg = missing,
                    reg_vec = missing,
                    penalty = missing)
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
        :p => missing,
        :reg => reg,
        :reg_vec => reg_vec,
        :penalty => penalty
      )
      return model
end
