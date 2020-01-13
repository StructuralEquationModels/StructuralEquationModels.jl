using RCall, Optim, LinearAlgebra, ForwardDiff, Random, NLSolversBase,
      Distributions, BenchmarkTools, DataFrames

# include functions from other scripts


# get Data from R
R"""
pacman::p_load(lavaan)

data(HolzingerSwineford1939)
dat <- HolzingerSwineford1939[7:9]
"""

dat = rcopy(R"dat")

#dat = convert(Matrix, dat)

# define model
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












###########################################################
x0 = append!([0.5, 0.5, 0.5, 0.5], ones(2))

@time fit_in_tree(ram, dat, x0, ML, "LBFGS")

@time fit_sem(ram, dat, x0, ML, "LBFGS")

fitted_lb = fit_sem(ram, dat, x0, ML, "LBFGS")

fitted_new = fit_sem(ram, dat, x0, ML, "Newton")

delta_method(fitted_lb)

delta_method(fitted_new)
