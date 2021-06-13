pacman::p_load(stringr)

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

gen_model_wol <- function(nfact, nitem){
  model <- c()
  for(i in 1:nfact){
    model[i] <- 
      str_c(
        "f", 
        i, 
        "=~", 
        str_sub(
          paste(str_c("x_", i, "_", 1:nitem, " + "), collapse = ""),
          end = -3),
        "\n ")
  }
  model <- paste(model, collapse = "")
  return(model)
}