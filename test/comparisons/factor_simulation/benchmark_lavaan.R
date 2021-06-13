pacman::p_load(lavaan, microbenchmark, dplyr)
setwd("C:/Users/maxim/.julia/dev/sem/test/comparisons/factor_simulation/")

data <- readr::read_rds("data.rds")

benchmarks <- 
  purrr::pmap_dfr(data, 
           ~with(
             list(...), 
             summary(microbenchmark(cfa(model, data), times = 2))))

benchmarks <- 
  bind_cols(
    select(data, -c(model, data)),
    select(benchmarks, -c(expr)))

readr::write_csv2(benchmarks, "benchmarks/benchmarks_lavaan.csv")
