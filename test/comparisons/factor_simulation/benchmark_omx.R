pacman::p_load(OpenMx, tidyverse, arrow, microbenchmark)

setwd("C:/Users/maxim/.julia/dev/sem/test/comparisons/factor_simulation/")
source("factor_functions.R")

mxOption(NULL,
         "Default optimizer",
         "NPSOL")

mxOption(NULL, "Calculate Hessian", "No")
mxOption(NULL, "Standard Errors", "No")

results <- readr::read_rds("data.rds")

results <- mutate(
  results,
  omx_model = 
    pmap(results, 
         ~with(
           list(...), 
           gen_model_omx(nfact_vec, nitem_vec, data)))
)

mxRun(model)

data = results$data[[1]]
nfact = 3
nitem = 5

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


benchmarks <- 
  purrr::pmap_dfr(
    results,
    ~with(list(...), 
          summary(microbenchmark(
            mxRun(omx_model), times = 2))))

benchmarks <- 
  bind_cols(
    select(results, -c(model, data, model_omx, )),
    select(benchmarks, -c(expr)))

readr::write_csv2(benchmarks, "benchmarks/benchmarks_omx.csv")