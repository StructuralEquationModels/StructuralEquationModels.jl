library(stringr)

lavaan_true_model <- function(
  n_factors,
  n_items,
  mean_load,
  sd_load,
  mean_cov,
  sd_cov,
  mean_mean,
  sd_mean,
  meanstructure){
  model <- c()
  for(i in 1:n_factors) {
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
  for (i in 1:n_factors) {
    for (j in 1:i) {
      if (i != j) {
        par <- rnorm(1, mean_cov, sd_cov)
        model <- append(model, str_c("f", i, "~~ ", par, "*f", j, "\n"))
      }else{
        model <- append(model, str_c("f", i, "~~ 1*f", j, "\n"))
      }
    }
  }
  if(meanstructure){
    means <- rnorm(n_items, mean_mean, sd_mean)
    for(i in 1:n_factors){
      for(j in 1:n_items){
        model <- append(model, str_c("x_", i, "_", j, " ~ " , means[j], "*1 \n"))
      }
    }
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
  if(meanstructure){
    for(i in 1:n_factors){
      for(j in 1:n_items){
        model <- append(model, str_c("x_", i, "_", j, " ~ " , letters[j], "*1 \n"))
      }
    }
  }
  model <- paste(model, collapse = "")
  return(model)
}

omx_model <- function(n_factors, n_items, data, meanstructure, lavpar){
  
  if(meanstructure){
    dataRaw <- mxData(observed=data, type="raw")
  }else{
    dataRaw <- mxData(observed=cov(data), type="cov", numObs = nrow(data))
  }
  
  nobs = n_factors*n_items
  lat_vars <- str_c("f", 1:n_factors)
  observed_vars <- str_c("x_", 1:n_factors, "_")
  observed_vars <- map(observed_vars, ~str_c(.x, 1:n_items))
  
  start_res <- 
    filter(
      lavpar, 
      lhs == rhs, 
      op == "~~",
      str_detect(lhs, "^x"))$start
  
  start_load <- 
    map(1:n_factors,
        ~filter(
          lavpar, 
          op == "=~", 
          lhs == str_c("f", .x)
          )$start)
  
  # residual variances
  resVars <- mxPath( from=unlist(observed_vars), arrows=2,
                     free=TRUE,
                     labels=str_c("e", 1:nobs),
                     values = start_res)
  # latent variances and covariance
  nvoc <- n_factors*(n_factors+1)/2
  
  latVars <- mxPath( lat_vars, arrows=2, connect="single",
                     free=FALSE, values = 1)
  latCov <- mxPath( lat_vars, arrows=2, connect="unique.bivariate",
                    free=TRUE, values = 0.0 )

  loadings <- pmap(list(lat_vars, observed_vars, start_load), 
                   ~mxPath(
                     from = ..1, 
                     to = ..2,
                     arrows = 1,
                     values = ..3,
                     free = rep(T, n_items),
                     labels = str_c("l_", .y)))
  
  # means
  if(meanstructure){
    means <- mxPath( from="one", c(unlist(observed_vars), lat_vars),
                     arrows=1,
                     free=c(rep(T, nobs), rep(F, n_factors)),
                     values=c(colMeans(data), rep(0, n_factors)) )
  }else{
    means <- NULL
  }
  
  #funML <- mxFitFunctionML()
  #expML <- mxExpectationNormal(covariance = "dataRaw")
  
  model <- 
    mxModel(
      str_c(
        "factor_model_n_factors_",
        n_factors,
        "_n_items_",
        n_items,
        "meanstructure_",
        meanstructure),
      type="RAM",
      manifestVars=observed_vars,
      latentVars=lat_vars,
      dataRaw, resVars, latVars, latCov, splice(loadings), means)
  return(model)
}

time_lavaan <- function(model, data, estimator){
  cfa(
    model,
    data,
    estimator = tolower(estimator),
    std.lv = TRUE,
    se = "none", test = "none",
    baseline = F, loglik = F, h1 = F)@timing$optim
}

time_omx <- function(model, estimator){
  if(tolower(estimator) != "ml"){
    return(NA)
  }else{
   time <- mxRun(model)@output$backendTime
  }
  return(time)
}

benchmark_lavaan <- function(model, data, n_repetitions, estimator){
  out <- 
    map_dfr(
      1:n_repetitions, 
      ~safe_and_quiet(
        time_lavaan,
        model,
        data,
        estimator)
    )
  return(out)
}

benchmark_omx <- function(model, n_repetitions, estimator){
  out <- 
    map_dfr(
      1:n_repetitions, 
      ~safe_and_quiet(
        time_omx,
        model,
        estimator)
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