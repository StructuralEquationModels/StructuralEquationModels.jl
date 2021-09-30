pacman::p_load(here, arrow, tidyverse, lavaan, microbenchmark, magrittr)

set.seed(123)

#setwd(r"(C:\Users\maxim\.julia\dev\sem)")

#----lavaan----
model_true <-    "# measurement model
            ind60 =~ x1 + 0.6*x2 + 0.5*x3
            dem60 =~ y1 + 0.8*y2 + 0.75*y3 + 0.87*y4
            dem65 =~ y5 + 0.4*y6 + 0.5*y7 + 0.7*y8
            # regressions
            dem60 ~ 0.4*ind60
            dem65 ~ 0.3*ind60 + 0.6*dem60
            x1 ~~ 0.7*x1
            x2 ~~ 0.8*x2
            x3 ~~ 0.3*x3
            y1 ~~ 0.5*y1
            y2 ~~ 0.54*y2
            y3 ~~ 0.776*y3
            y4 ~~ 0.46*y4
            y5 ~~ 0.87*y5
            y6 ~~ 0.7*y6
            y7 ~~ 0.76*y7
            y8 ~~ 0.87*y8"

model <-    "# measurement model
            ind60 =~ x1 + x2 + x3
            dem60 =~ y1 + y2 + y3 + y4
            dem65 =~ y5 + y6 + y7 + y8
            # regressions
            dem60 ~ ind60
            dem65 ~ ind60 + dem60"

data <- lavaan::simulateData(model_true)

fit_ml <- cfa(model, data, likelihood = "wishart", do.fit = FALSE)
fit_ls <- cfa(model, data, estimator = "WLS", do.fit = FALSE)#,  WLS.V = diag(66))

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

write_arrow(par_ml, str_c("test/comparisons/par_sem_ml.arrow"))
write_arrow(par_ls, str_c("test/comparisons/par_sem_ls.arrow"))
write_arrow(data, str_c("test/comparisons/data_sem.arrow"))
