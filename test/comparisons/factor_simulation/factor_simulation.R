pacman::p_load(lavaan, tidyverse, arrow)

setwd("C:/Users/maxim/.julia/dev/sem/test/comparisons/factor_simulation/")

config <- read_csv2("config_factor.csv")

source("factor_functions.R")

results <- config

results <- mutate(
  results,
  model = pmap_chr(results,  ~gen_model(.x, .y, 0.5, 0.2)))

results <- mutate(
  results,
  data = map(model, ~simulateData(.x, sample.nobs = 5000)))

results <- mutate(
  results,
  model = pmap_chr(results,  ~gen_model_wol(.x, .y)))

pwalk(results, 
      ~with(
        list(...), 
        arrow::write_arrow(
          data, 
          str_c(
            "data/",
            "nfact_",
            nfact_vec,
            "_nitem_",
            nitem_vec,
            ".arrow")
        )
        )
      )

write_rds(results, "data.rds")
