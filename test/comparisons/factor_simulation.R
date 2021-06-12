pacman::p_load(lavaan, tidyverse, arrow)

nfact_vec <- c(3, 5)

nitem_vec <- c(5, 10, 20, 48)

gen_model <- function(nfact, nitem, mean_load, sd_load){
  model <- c()
  for(i in 1:nfact){
    load <- rnorm(nitem, mean_load, sd_load)
    model[i] <- 
      str_c(
      "f", 
      i, 
      "=~", 
      str_sub(
        paste(str_c(load, "*x_", i, "_", 1:nitem, " + "), collapse = ""),
        end = -3),
      "\n ")
  }
  model <- paste(model, collapse = "")
  return(model)
}

results <- expand_grid(nfact_vec, nitem_vec)

results <- mutate(
  results,
  model = pmap_chr(results,  ~gen_model(.x, .y, 0.5, 0.2)))

results <- mutate(
  results,
  data = map(model, ~simulateData(.x, sample.nobs = 5000)))

# results <- mutate(
#   results,
#   lav_fit = map2(model, data, ~cfa(.x, .y))
# )

pwalk(results, 
      ~with(
        list(...), 
        arrow::write_arrow(
          data, 
          str_c(
            "C:/Users/maxim/.julia/dev/sem/test/comparisons/factor_simulation/",
            "nfact_",
            nfact_vec,
            "_nitem_",
            nitem_vec,
            ".arrow")
        )
        )
      )