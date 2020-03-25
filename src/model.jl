function model(ram, data, par, est = ML, optim = "LBFGS")
    data_matr = convert(Matrix{Float64}, data)

    model = Dict{Symbol, Any}(
        :ram => ram,
        :par => par,
        :data => data_matr,
        :optim => optim,
        :obs_cov => missing,
        :imp_cov => missing,
        :obs_mean => missing,
        :logl => missing,
        :opt_result => missing
      )
      return model
end

function ram(x)
      S =   [x[1] 0 0 0
            0 x[2] 0 0
            0 0 x[3] 0
            0 0 0 x[4]]

      F =  [1 0 0 0
            0 1 0 0
            0 0 1 0]

      A =  [0 0 0 1
            0 0 0 x[5]
            0 0 0 x[6]
            0 0 0 0]

      return (S, F, A)
end
