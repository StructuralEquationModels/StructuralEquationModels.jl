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

gen_model_omx <- function(nfact, nitem, data){
  
  dataRaw <- mxData( observed=data, type="raw" )
  
  nobs = nfact*nitem
  lat_vars <- str_c("f", 1:nfact)
  observed_vars <- str_c("x_", 1:nfact, "_")
  observed_vars <- map(observed_vars, ~str_c(.x, 1:nitem))
  
  # residual variances
  resVars <- mxPath( from=unlist(observed_vars), arrows=2,
                     free=TRUE,
                     labels=str_c("e", 1:nobs) )
  # latent variances and covariance
  latVars <- mxPath( lat_vars, arrows=2, connect="unique.pairs",
                     free=TRUE)
  
  loadings <- map2(lat_vars, observed_vars, 
                   ~mxPath(
                     from = .x, 
                     to = .y,
                     arrows = 1,
                     values = rep(1, nitem),
                     free = c(F, rep(T, nitem-1)),
                     labels = str_c("l_", .y)))
  
  # means
  means <- mxPath( from="one", c(unlist(observed_vars), lat_vars),
                   arrows=1,
                   free=c(rep(T, nobs), rep(F, nfact)), 
                   values=c(rep(1, nobs), rep(0, nfact)) )
  
  model <- 
    mxModel(
      str_c("factor_model_nfact_", nfact, "_nitem_", nitem),
      type="RAM",
      manifestVars=observed_vars,
      latentVars=lat_vars,
      dataRaw, resVars, latVars, splice(loadings), means)
  return(model)
}