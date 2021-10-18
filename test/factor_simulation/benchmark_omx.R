pacman::p_load(OpenMx, tidyverse, arrow, microbenchmark)

setwd("C:/Users/maxim/.julia/dev/sem/test/comparisons/factor_simulation/")
source("factor_functions.R")

mxOption(NULL, "Default optimizer", "NPSOL")
mxOption(NULL, "Calculate Hessian", "No")
mxOption(NULL, "Standard Errors", "No")

results <- readr::read_rds("data.rds")

results <- mutate(
  results,
  omx_model = 
    pmap(results, 
         ~with(
           list(...), 
           gen_model_omx(nfact_vec, nitem_vec, data, starting_values$est)))
)

benchmarks <- 
  purrr::pmap_dfr(
    results,
    ~with(list(...), 
          summary(microbenchmark(
            mxRun(omx_model), times = 1, unit = "s"))))

benchmarks <- 
  bind_cols(
    select(results, nfact_vec, nitem_vec, nobs),
    select(benchmarks, -c(expr)))

readr::write_csv2(benchmarks, "benchmarks/benchmarks_omx.csv")
