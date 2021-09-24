pacman::p_load(here, arrow, tidyverse, lavaan, microbenchmark, magrittr)

set.seed(123)

#setwd(r"(C:\Users\maxim\.julia\dev\sem)")

#----lavaan----
model <- '
  # intercept and slope with fixed coefficients
    i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
    s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4
  # regressions
    i ~ x1 + x2
    s ~ x1 + x2
  # time-varying covariates
    t1 ~ c1
    t2 ~ c2
    t3 ~ c3
    t4 ~ c4
'

data <- Demo.growth

fit_ml <- cfa(model, data, likelihood = "wishart", do.fit = FALSE)
fit_ls <- cfa(model, data, estimator = "WLS", do.fit = FALSE)

V <- lavInspect(fit_ls, "WLS.V")
#V <- V[-c(1:11), -c(1:11)]

obs_cov <- cov(data)
obs_cov <- solve(obs_cov)
obs_cov <- kronecker(obs_cov, obs_cov)
L <- lavaan::lav_matrix_duplication(11)
W <- t(L)%*%obs_cov%*%L

K = L%*%solve(t(L)%*%L)

fit_ml <- cfa(model, data, likelihood = "wishart", do.fit = TRUE)
fit_ls <- cfa(model, data, estimator = "WLS", do.fit = TRUE, WLS.V = diag(55))

par_ml <- select(parTable(fit_ml), lhs, op, rhs, est, start)
par_ls <- select(parTable(fit_ls), lhs, op, rhs, est, start)

write_arrow(par_ml, str_c("test/comparisons/par_growth_ml.arrow"))
write_arrow(par_ls, str_c("test/comparisons/par_growth_ls.arrow"))
write_arrow(data, str_c("test/comparisons/data_growth.arrow"))
