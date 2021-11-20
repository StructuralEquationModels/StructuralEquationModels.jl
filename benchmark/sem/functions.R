pacman::p_load(stringr)

lavaan_true_model <- function(
  n_factors, 
  n_items, 
  mean_load, 
  sd_load,
  mean_reg,
  sd_reg){
  model <- c()
  for(i in 1:n_factors){
    load <- rnorm(n_items, mean_load, sd_load)
    model[i] <- 
      str_c(
        "f", 
        i, 
        "=~", 
        str_sub(
          paste(str_c(load, "*x_", i, "_", 1:n_items, " + "), collapse = ""),
          end = -3),
        "\n ")
  }
  for(i in 1:n_factors){
    for(j in 1:n_items){
      item <- str_c("x_", i, "_", j)
      model <- append(model, str_c(item, "~~ 0.25*", item, " \n"))
    }
  }
  for(i in 1:n_factors){
    model <- append(model, str_c("f", i, "~~ 1*f", i, "\n"))
  }
  for(i in 1:(n_factors-1)){
      par <- rnorm(1, mean_reg, sd_reg)
      model <- append(model, str_c("f", i+1, "~ ", par, "*f", i, "\n"))
  }
  model <- paste(model, collapse = "")
  return(model)
}

lavaan_model <- function(n_factors, n_items, meanstructure){
  model <- c()
  for(i in 1:n_factors){
    model[i] <- 
      str_c(
        "f", 
        i, 
        "=~", 
        str_sub(
          paste(str_c("x_", i, "_", 1:n_items, " + "), collapse = ""),
          end = -3),
        "\n ")
  }
  for(i in 1:(n_factors-1)){
    model <- append(model, str_c("f", i+1, "~f", i, "\n"))
  }
  model <- paste(model, collapse = "")
  return(model)
}

omx_model <- function(n_factors, n_items, data){
  
  dataRaw <- mxData(observed=data, type="raw")
  
  nobs = n_factors*n_items
  lat_vars <- str_c("f", 1:n_factors)
  observed_vars <- str_c("x_", 1:n_factors, "_")
  observed_vars <- map(observed_vars, ~str_c(.x, 1:n_items))
  
  lat_from <- str_c("f", 1:(n_factors-1))
  lat_to <- str_c("f", 2:n_factors)
  
  # residual variances
  resVars <- mxPath( from=unlist(observed_vars), arrows=2,
                     free=TRUE,
                     labels=str_c("e", 1:nobs),
                     values = 1)
  # latent variances and covariance
  latVars <- mxPath( lat_vars, arrows=2, connect="single",
                     free=FALSE, values = 1)
  
  latCov <- mxPath( lat_from, lat_to, arrows=1, connect="single",
                    free=TRUE, values = 0)

  loadings <- pmap(list(lat_vars, observed_vars), 
                   ~mxPath(
                     from = ..1, 
                     to = ..2,
                     arrows = 1,
                     values = 1,
                     free = rep(T, n_items),
                     labels = str_c("l_", .y)))
  
  means <- mxPath(from = 'one', to = unlist(observed_vars), 
                  arrows=1,
                  free=TRUE,
                  values = 0)
  
  model <- 
    mxModel(
      str_c(
        "factor_model_n_factors_",
        n_factors,
        "_n_items_",
        n_items),
      type="RAM",
      manifestVars=observed_vars,
      latentVars=lat_vars,
      dataRaw, resVars, latVars, latCov, splice(loadings), means)
  return(model)
}

time_lavaan <- function(model, data){
  sem(
    model,
    data,
    std.lv = TRUE,
    se = "none", test = "none",
    missing = "fiml",
    start = "simple",
    baseline = F, loglik = F, h1 = F)@timing$optim
}

benchmark_lavaan <- function(model, data, n_repetitions){
  out <- 
    map_dfr(
      1:n_repetitions, 
      ~safe_and_quiet(
        time_lavaan,
        model,
        data)
      )
  return(out)
}

time_omx <- function(model){
  mxRun(model)@output$backendTime
}

benchmark_omx <- function(model, n_repetitions){
  out <- 
    map_dfr(
      1:n_repetitions, 
      ~safe_and_quiet(
        time_omx,
        model)
    )
  return(out)
}

safe_and_quiet <- function(fun, ...){
  safe_fun <- quietly(safely(fun))
  out_safe <- safe_fun(...)
  out <- 
    list(
      result = out_safe$result$result,
      error = out_safe$result$error,
      output = out_safe$output,
      warnings = out_safe$warnings,
      messages = out_safe$messages)
  if(!is.null(out$error)){
    out$error <- conditionMessage(out$error)
  }
  out <- map(out, null_to_na)
  return(out)
}

null_to_na <- function(obj){
  if(is.null(obj)|(length(obj)==0)){
    return(NA)
  }else{
    return(obj)
  }
}

extract_results <- function(df){
  mean_time <- mean(as.double(df$result, units = "secs"), na.rm = TRUE)
  median_time <- median(as.double(df$result, units = "secs"), na.rm = TRUE)
  sd_time <- sd(as.double(df$result, units = "secs"), na.rm = TRUE)
  error = unique(df$error)
  warnings = unique(df$warnings)
  messages = unique(df$messages)
  return(list(
    mean_time = mean_time,
    median_time = median_time,
    sd_time = sd_time,
    error = error,
    warnings = warnings,
    messages = messages
  ))
}

induce_missing <- function(data, p){
  data <- 
    mutate(
      data, 
      across(
        everything(),
        ~ifelse(rbinom(length(.x), 1, p), NA, .x)
        )
      )
  return(data)
}
