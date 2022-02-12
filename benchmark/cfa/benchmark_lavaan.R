library(lavaan)
library(dplyr)
library(purrr)
library(readr)

set.seed(73647820)
setwd("./benchmark/cfa")
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
         start =
           pmap(
             results,
             ~with(
               list(...),
               cfa(model_lavaan,
                 data,
                 estimator = "ml",
                 std.lv = TRUE,
                 do.fit = FALSE))))

results <-
  mutate(results,
         start =
           map(
             start,
             parTable))

const <- 3*(results$n_par[length(results$n_par)]^2)

results <- mutate(
  results,
  n_repetitions = round(const/(n_par^2)))

#!!!
# results$n_repetitions <- 10

results <- filter(results, meanstructure == 0)
##

benchmarks <- pmap(
  results, 
  ~with(list(...),
        benchmark_lavaan(
          model_lavaan, 
          data, 
          n_repetitions, 
          Estimator)
        )
  )

benchmark_summary <- map_dfr(benchmarks, extract_results)
benchmark_summary <- rename_with(benchmark_summary, ~str_c(.x, "_lav"))

results <- bind_cols(results, benchmark_summary)

write_csv2(select(
  results, 
  Estimator, 
  n_factors, 
  n_items, 
  meanstructure,
  n_repetitions,
  n_obs,
  mean_time_lav,
  median_time_lav,
  sd_time_lav,
  error_lav,
  warnings_lav,
  messages_lav), "results/benchmarks_lavaan.csv")

write_rds(results, "results.rds")