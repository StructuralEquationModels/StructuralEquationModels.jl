pacman::p_load(lavaan, dplyr, purrr, readr)
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

const <- 3*(results$n_par[length(results$n_par)]^2)

results <- mutate(
  results,
  n_repetitions = round(const/(n_par^2)))

#!!!
results$n_repetitions <- 2
##

benchmarks <- pmap(
  results, 
  ~with(list(...),
        benchmark_lavaan(
          model_lavaan, 
          data, 
          n_repetitions)
        )
  )

benchmark_summary <- map_dfr(benchmarks, extract_results)
benchmark_summary <- rename_with(benchmark_summary, ~str_c(.x, "_lav"))

results <- bind_cols(results, benchmark_summary)

results %>%
  ggplot(aes(
    x = n_factors * n_items,
    y = mean_time_lav,
    color = as.factor(missingness)
  )) +
  geom_line() +
  geom_point() +
  theme_minimal()
  

write_csv2(select(
  results, 
  Estimator, 
  n_factors, 
  n_items, 
  missingness,
  n_repetitions,
  n_obs,
  mean_time_lav,
  median_time_lav,
  sd_time_lav,
  error_lav,
  warnings_lav,
  messages_lav), "results/benchmarks_lavaan.csv")

write_rds(results, "results.rds")
