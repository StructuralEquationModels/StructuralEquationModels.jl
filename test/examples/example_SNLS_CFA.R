pacman::p_load(here, arrow, tidyverse, lavaan, microbenchmark, magrittr)

set.seed(123)

#setwd(r"(C:\Users\maxim\.julia\dev\sem)")

#----lavaan----
model <-    "visual =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed =~ x7 + x8 + x9
            x7 ~~ x8"

data <- HolzingerSwineford1939
data <- select(data, starts_with("x"))

#fit_ml <- cfa(model, data, likelihood = "wishart", do.fit = FALSE)
#fit_ls <- cfa(model, data, estimator = "WLS", do.fit = FALSE,  WLS.V = diag(45))

#V <- lavInspect(fit_ls, "WLS.V")
#V <- V[-c(1:11), -c(1:11)]

#obs_cov <- cov(data)
#obs_cov <- solve(obs_cov)
#obs_cov <- kronecker(obs_cov, obs_cov)
#L <- lavaan::lav_matrix_duplication(11)
#W <- t(L)%*%obs_cov%*%L

#K = L%*%solve(t(L)%*%L)

fit_ml <- cfa(model, data, likelihood = "wishart", do.fit = TRUE)
fit_ls <- cfa(model, data, estimator = "WLS", do.fit = TRUE, WLS.V = diag(45))

par_ml <- select(parTable(fit_ml), lhs, op, rhs, est, start)
par_ls <- select(parTable(fit_ls), lhs, op, rhs, est, start)

write_arrow(par_ml, str_c("test/comparisons/par_hol_ml.arrow"))
write_arrow(par_ls, str_c("test/comparisons/par_hol_ls.arrow"))
write_arrow(data, str_c("test/comparisons/data_hol.arrow"))
