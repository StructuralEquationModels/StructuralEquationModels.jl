pacman::p_load(here, arrow, tidyverse, lavaan, microbenchmark, magrittr)

set.seed(123)

setwd(r"(C:\Users\maxim\.julia\dev\sem)")

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

data <- select(data, starts_with("x"), starts_with("y"))

fit_ml <- cfa(model, data, likelihood = "wishart", do.fit = TRUE)
fit_ls <- cfa(model, data, estimator = "GLS", do.fit = TRUE)

# timing
times <- map_dbl(
  1:100, ~cfa(model, data, likelihood = "wishart", do.fit = TRUE)@timing$optim)
microbenchmark(cfa(model, data, likelihood = "wishart", do.fit = TRUE, se = "none", test = "none",
                   baseline = F, loglik = F, h1 = F))

times <- map_dbl(1:100, ~cfa(model, data, estimator = "GLS", do.fit = TRUE)@timing$optim)


par_ml <- select(parTable(fit_ml), lhs, op, rhs, est, start)
par_ls <- select(parTable(fit_ls), lhs, op, rhs, est, start)

write_arrow(par_ml, str_c("test/comparisons/par_dem_ml.arrow"))
write_arrow(par_ls, str_c("test/comparisons/par_dem_ls.arrow"))
write_arrow(data, str_c("test/comparisons/data_dem.arrow"))


# bootstrap samples -------------------------------------------------------

n = 100
n_obs = nrow(data)
n_obs_gen = 50
max_iter = 200

bootstrap_samples <- function(data, nobs, n_obs_gen, n){
  obs <- 1:nobs
  rowind <- map(1:n, ~sample(obs, n_obs_gen))
  samples <- map(rowind, ~data[.x, ])
  return(samples)
}

data_boot <- bootstrap_samples(data, n_obs, n_obs_gen, n)

fits_ml <- map(data_boot, ~cfa(model, .x, likelihood = "wishart", do.fit = TRUE))
fits_ls <- map(data_boot, ~cfa(model, .x, estimator = "GLS", do.fit = TRUE))

sum(map_lgl(fits_ml, ~lavInspect(.x, "converged")))

sum(map_lgl(fits_ls, ~lavInspect(.x, "converged")))


library(microbenchmark)

microbenchmark(cfa(model, data, likelihood = "wishart", do.fit = TRUE))

