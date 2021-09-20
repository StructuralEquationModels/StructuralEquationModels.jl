pacman::p_load(here, arrow, tidyverse, lavaan, microbenchmark, magrittr)

set.seed(123)

#setwd(r"(C:\Users\maxim\.julia\dev\sem)")

#----lavaan----
model <-    "# measurement model
            ind60 =~ x1 + x2 + x3
            dem60 =~ y1 + y2 + y3 + y4
            dem65 =~ y5 + y6 + y7 + y8
            # regressions
            dem60 ~ ind60
            dem65 ~ ind60 + dem60
            # residual correlations
            y1 ~~ y5
            y2 ~~ y4 + y6
            y3 ~~ y7
            y4 ~~ y8
            y6 ~~ y8"

data <- PoliticalDemocracy

fit_ml <- cfa(model, data, likelihood = "wishart", do.fit = FALSE)
fit_ls <- cfa(model, data, estimator = "WLS", do.fit = FALSE,  WLS.V = diag(66))

V <- lavInspect(fit_ls, "WLS.V")
#V <- V[-c(1:11), -c(1:11)]

obs_cov <- cov(data)
obs_cov <- solve(obs_cov)
obs_cov <- kronecker(obs_cov, obs_cov)
L <- lavaan::lav_matrix_duplication(11)
W <- t(L)%*%obs_cov%*%L

K = L%*%solve(t(L)%*%L)

fit_ml <- cfa(model, data, likelihood = "wishart", do.fit = TRUE)
fit_ls <- cfa(model, data, estimator = "WLS", do.fit = TRUE, WLS.V = diag(66))

par_ml <- select(parTable(fit_ml), lhs, op, rhs, est, start)
par_ls <- select(parTable(fit_ls), lhs, op, rhs, est, start)

write_arrow(par_ml, str_c("test/comparisons/par_dem_ml.arrow"))
write_arrow(par_ls, str_c("test/comparisons/par_dem_ls.arrow"))
write_arrow(data, str_c("test/comparisons/data_dem.arrow"))