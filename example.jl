using RCall, Optim, LinearAlgebra, ForwardDiff, Random, NLSolversBase,
      Distributions, BenchmarkTools, DataFrames

# include functions from other scripts
include("objective_functions.jl")
include("helper_functions.jl")
include("sem_wrapper_functions.jl")

# get Data from R
R"""
pacman::p_load(lavaan)

data(HolzingerSwineford1939)
dat <- HolzingerSwineford1939[7:9]
"""

dat = rcopy(R"dat")

#dat = convert(Matrix, dat)

# define model as the researcher should do.
# Note: Variables are currently assessed by position!
# DataFrame is converted to Matrix internally.
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


x0 = append!([0.5, 0.5, 0.5, 0.5], ones(2))

fit_in_tree(ram, dat, x0, ML, "LBFGS")

fit_sem(ram, dat, x0, ML, "LBFGS")

fitted_lb = fit_sem(ram, dat, x0, ML, "LBFGS")

fitted_new = fit_sem(ram, dat, x0, ML, "Newton")

delta_method(fitted_lb)

delta_method(fitted_new)

############### example 2: 3 factors and latent correlations

R"""
pacman::p_load(lavaan)

data(HolzingerSwineford1939)
dat2 <- HolzingerSwineford1939
"""

dat2 = rcopy(R"dat2")

dat2[7:15]

x = ones(6)

S =   :([x[1] 0 0 0
      0 x[2] 0 0
      0 0 x[3] 0
      0 0 0 x[4]])

F =  :([1 0 0 0
      0 1 0 0
      0 0 1 0])

A =  :([0 0 0 1
      0 0 0 x[5]
      0 0 0 x[6]
      0 0 0 0])



function ram(x, S, F, A)
      x = :($x)
      S = eval(S)
      F = eval(F)
      A = eval(A)
      return (S, F, A)
      #return x
end

S =   :([:($x)[1] 0 0 0
      0 :($x)[2] 0 0
      0 0 :($x)[3] 0
      0 0 0 :($x)[4]])

F =  :([1 0 0 0
      0 1 0 0
      0 0 1 0])

A =  :([0 0 0 1
      0 0 0 :($x)[5]
      0 0 0 :($x)[6]
      0 0 0 0])

function ram(x, S, F, A)
      x = :($x)
      S =   [:($x)[1] 0 0 0
            0 :($x)[2] 0 0
            0 0 :($x)[3] 0
            0 0 0 :($x)[4]]

      F =  [1 0 0 0
            0 1 0 0
            0 0 1 0]

      A =  [0 0 0 1
            0 0 0 :($x)[5]
            0 0 0 :($x)[6]
            0 0 0 0]
      return (S, F, A)
end

function ramram(x, S, F, A)
      x = x
      return ram(x, S, F, A)
end

ramram(zeros(6), S, F, A)

function sum2(a, b)
      return a+b
end

@parallel sum2(5,5)
