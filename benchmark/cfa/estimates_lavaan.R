library(lavaan)
library(dplyr)
library(purrr)
library(readr)

set.seed(73647820)
source("functions.R")

results <- readr::read_rds("results.rds")

results <-
  mutate(results,
         model_lavaan =
           pmap_chr(
             results,
             ~with(
               list(...),
               lavaan_model(n_factors, n_items, meanstructure))))

# results$model_lavaan[[24]] <- str_remove_all(results$model_lavaan[[24]], "NA")
results$model_lavaan[[12]] <- str_remove_all(results$model_lavaan[[12]], "NA")

results <-
  mutate(results,
         estimate =
           pmap(
             results,
             ~with(
               list(...),
               cfa(
                model_lavaan,
                data,
                estimator = tolower(Estimator),
                std.lv = TRUE,
                se = "none", test = "none",
                baseline = F, loglik = F, h1 = F))))

results <-
  mutate(results,
         estimate =
           map(
             estimate,
             parTable))

pwalk(results, 
      ~with(
        list(...), 
        write_csv(
          estimate, 
          str_c(
            "parest/",
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
