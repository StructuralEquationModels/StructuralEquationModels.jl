pacman::p_load(lavaan, readr, dplyr, purrr)
set.seed(193273)
source("functions.R")
config <- read_csv2("config.csv")

results <- config

results <- mutate(
  results,
  model = pmap_chr(
    results,
    ~with(list(...), lavaan_true_model(
      n_factors,
      n_items,
      0.5, 0.2, 0.3, 0.1)
    )
  )
)

results <- mutate(
  results,
  n_par = map2_dbl(
    n_factors,
    n_items,
    ~ 3*(.x*.y) + .x-1
    )
  )

results <- mutate(
  results,
  n_obs = 25*n_par
  )

data <- mutate(
  filter(results, missingness == 0),
  data = pmap(
  filter(results, missingness == 0),
  ~ with(
    list(...),
    simulateData(model,
                 sample.nobs = n_obs,
                 std.lv = TRUE)
  ))
)

results <- full_join(
  results,
  select(data, n_factors, n_items, data),
  by = c("n_factors", "n_items"))

results <- mutate(
  results,
  data = map2(data, missingness, ~induce_missing(.x, .y))
)

# write data to disk ------------------------------------------------------

pwalk(results, 
      ~with(
        list(...), 
        write_csv(
          data, 
          str_c(
            "data/",
            "n_factors_",
            n_factors,
            "_n_items_",
            n_items,
            "_missing_",
            missingness,
            ".csv")
        )
        )
      )

write_rds(results, "results.rds")
