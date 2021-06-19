pacman::p_load(lavaan, microbenchmark, dplyr)
setwd("C:/Users/maxim/.julia/dev/sem/test/comparisons/factor_simulation/")

data <- readr::read_rds("data.rds")

benchmarks <- 
  purrr::pmap_dfr(data, 
           ~with(
             list(...), 
             summary(microbenchmark(
               cfa(model, data, meanstructure = TRUE, 
                   missing = "fiml", orthogonal = TRUE,
                   std.lv = TRUE), times = 1, 
               unit = "s"))))

benchmarks <- 
  bind_cols(
    select(results, nfact_vec, nitem_vec, nobs),
    select(benchmarks, -c(expr)))

readr::write_csv2(benchmarks, "benchmarks/benchmarks_lavaan.csv")
