setwd("C:/Users/maxim/.julia/dev/sem/test/comparisons/factor_simulation/")

nfact_vec <- c(3, 5)

nitem_vec <- c(10)#, 5, 20, 48)

nobs <- c(500)

p_missing <- c(0.1)

config <- tidyr::expand_grid(nfact_vec, nitem_vec, nobs, p_missing)

readr::write_csv2(config, "config_factor.csv")
