pacman::p_load(OpenMx, dplyr, purrr, readr, umx, microbenchmark)
set.seed(85425301)
source("functions.R")

mxOption(NULL, "Default optimizer", "CSOLNP")
mxOption(NULL, "Calculate Hessian", "No")
mxOption(NULL, "Standard Errors", "No")
mxOption(NULL, "Number of Threads", omxDetectCores() - 1)

results <- readr::read_rds("results.rds")

results <- mutate(
  results,
  omx_model = 
    pmap(results, 
         ~with(
           list(...), 
           omx_model(n_factors, n_items, data, meanstructure, start)))
)

lavpar <- results$start[[7]]
filter(lavpar, op == "~1")

fit <- mxRun(results$omx_model[[6]])
fit@output$frontendTime
fit@output$backendTime
fit@output$evaluations

fit_lav <- cfa(
  results$model_lavaan[[1]],
  results$data[[1]],
  estimator = "ml",
  std.lv = TRUE,
  se = "none", test = "none",
  baseline = F, loglik = F, h1 = F)

benchmarks <- 
  purrr::pmap_dfr(
    filter(results, Estimator == "ML"),
    ~with(list(...), 
          summary(microbenchmark(
            mxRun(omx_model), times = 1, unit = "s"))))

benchmarks <- 
  bind_cols(
    select(results, nfact_vec, nitem_vec, nobs),
    select(benchmarks, -c(expr)))

readr::write_csv2(benchmarks, "benchmarks/benchmarks_omx.csv")
