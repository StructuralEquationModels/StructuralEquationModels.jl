library(lavaan)
library(readr)
library(dplyr)
library(purrr)

set.seed(193273)
setwd("./benchmark/cfa")
source("functions.R")
config <- read_csv("config.csv")

results <- config

results <- mutate(
  results,
  model = pmap_chr(
    results,
    ~with(list(...), lavaan_true_model(
      n_factors,
      n_items,
      0.5, 0.2, 0.3, 0.1, 0.5, 0.3,
      meanstructure)
    )
  )
)

results <- mutate(
  results,
  n_par = map2_dbl(
    n_factors,
    n_items,
    ~ 2*(.x*.y) + .x*(.x-1)/2
    )
  )

results <- mutate(
  results,
  n_obs = 25*n_par
  )

results <- mutate(results,
                  data = pmap(results,
                              ~ with(
                                list(...),
                                simulateData(model,
                                             sample.nobs = n_obs,
                                             std.lv = TRUE)
                              )))

# write data to disk ------------------------------------------------------

pwalk(results, 
      ~with(
        list(...), 
        write_csv(
          data, 
          str_c(
            "data/",
            "n_factors_",
            n_factors,
            "_n_items_",
            n_items,
            "_meanstructure_",
            meanstructure,
            ".csv")
        )
        )
      )

write_rds(results, "results.rds")
