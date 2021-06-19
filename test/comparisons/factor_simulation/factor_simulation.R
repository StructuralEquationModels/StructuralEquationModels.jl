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
  data = pmap(results, 
              ~with(list(...),
                    simulateData(model, 
                                 sample.nobs = nobs,
                                 orthogonal = TRUE,
                                 std.lv = TRUE))))

results <- mutate(
  results,
  data = map(data,  ~induce_missing(.x, 0.1)))

results <- mutate(
  results,
  model = pmap_chr(results,  ~gen_model_wol(.x, .y)))

####################### generate parameter estimates
results <- mutate(
  results,
  starting_values = 
    pmap(results,
         ~with(list(...),
               parameterEstimates(cfa(model, data, meanstructure = TRUE,
                   missing = "fiml", orthogonal = TRUE, do.fit = FALSE,
                   std.lv = TRUE)))))

results <- mutate(
  results,
  fits = pmap(results, ~with(list(...),
                             cfa(model, data, meanstructure = TRUE,
                                 missing = "fiml", orthogonal = TRUE,
                                 std.lv = TRUE))))

results <- mutate(
  results,
  parest = pmap(results, ~with(list(...), parameterEstimates(fits))))

pwalk(results,
      ~with(
        list(...),
        arrow::write_arrow(
          parest,
          str_c(
            "parest/",
            "nfact_",
            nfact_vec,
            "_nitem_",
            nitem_vec,
            ".arrow")
        )
      )
)

pwalk(results,
      ~with(
        list(...),
        arrow::write_arrow(
          starting_values,
          str_c(
            "start/",
            "nfact_",
            nfact_vec,
            "_nitem_",
            nitem_vec,
            ".arrow")
        )
      )
)
#######################

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
