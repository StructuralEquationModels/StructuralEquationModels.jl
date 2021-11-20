pacman::p_load(OpenMx, dplyr, purrr, readr, umx, microbenchmark, lubridate)
set.seed(85425301)
source("functions.R")

mxOption(NULL, "Default optimizer", "NPSOL")
mxOption(NULL, "Calculate Hessian", "No")
mxOption(NULL, "Standard Errors", "No")
mxOption(NULL, "Number of Threads", omxDetectCores() - 1)
mxOption(NULL, "Analytic Gradients", "Yes")

results <- readr::read_rds("results.rds")

results <- mutate(
  results,
  model_omx = 
    pmap(results, 
         ~with(
           list(...), 
           omx_model(n_factors, n_items, data)))
)

benchmarks <- pmap(
  results, 
  ~with(list(...),
        benchmark_omx(
          model_omx, 
          n_repetitions)
  )
)

benchmark_summary <- map_dfr(benchmarks, extract_results)
benchmark_summary <- rename_with(benchmark_summary, ~str_c(.x, "_omx"))

results <- bind_cols(results, benchmark_summary)

write_csv2(select(
  results, 
  Estimator, 
  n_factors, 
  n_items,
  missingness,
  n_repetitions,
  n_obs,
  mean_time_omx,
  median_time_omx,
  sd_time_omx,
  error_omx,
  warnings_omx,
  messages_omx), "results/benchmarks_omx.csv")

write_rds(results, "results.rds")
write_rds(benchmarks, "results/benchmarks_omx.rds")


fit_omx <- mxRun(results$model_omx[[1]])
fit_lav <- 
  sem(
    results$model_lavaan[[1]], 
    results$data[[1]], 
    missing = "fiml",
    std.lv = TRUE)
