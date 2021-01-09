pacman::p_load(here, feather, tidyverse, lavaan, microbenchmark, magrittr)

set.seed(123)

#setwd(r"(C:\Users\maxim\.julia\dev\sem)")

induce_missing <- function(v, p){
  miss <- sample(c(0,1), length(v), replace = TRUE, prob = c(1-p, p))
  ifelse(miss, NA, v)
}

get_testpars <- function(fit) {
  select(parameterEstimates(fit), lhs, op, rhs, est, se, p = pvalue, z)
}

# Definition Variables ----------------------------------------------------

library(OpenMx)

#model <- ' i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
#s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4 '
#growth_fit <- growth(model, data=Demo.growth)
#summary(growth_fit)


# small model -------------------------------------------------------------

data_small <- data.frame(load_t1 = rep(1, 100))

for(i in 1:4){
  name <- str_c("load_t", i)
  data_small[[name]] <- i
}

data_small %<>% mutate(across(contains("load"), function(x)x+rnorm(100, 0, 0.5)))

#dataRaw <- mxData( observed=Demo.growth, type="raw" )
# residual variances
resVars <- mxPath( from=c("t1","t2","t3","t4"), arrows=2,
                   free=TRUE, values = c(1,1,1,1))
# latent variances and covariance
latVars <- mxPath( from=c("intercept","slope"), arrows=2, connect="unique.pairs",
                   free=TRUE, values=c(1,1,1), labels=c("vari","cov","vars") )
# intercept loadings
intLoads <- mxPath( from="intercept", to=c("t1","t2","t3","t4"), arrows=1,
                    free=FALSE, values=c(1,1,1,1) )
# slope loadings
sloLoads <- mxPath( from="slope", to=c("t1","t2","t3","t4"), arrows=1,
                    free=FALSE, #values=c(0,1,2,3), 
                    labels = c("data.load_t1",
                               "data.load_t2",
                               "data.load_t3",
                               "data.load_t4"))
# manifest means
manMeans <- mxPath( from="one", to=c("t1","t2","t3","t4"), arrows=1,
                    free=FALSE, values=c(0,0,0,0) )
# latent means
latMeans <- mxPath( from="one", to=c("intercept", "slope"), arrows=1,
                    free=TRUE, values=c(1,1), labels=c("meani","means") )


dataRaw_small <- mxData( observed=data_small, type="raw" )

growthCurveModel_small <- mxModel("Linear Growth Curve Model Path Specification", 
                                   type="RAM",
                                   manifestVars=c("t1","t2","t3","t4"),
                                   latentVars=c("intercept","slope"),
                                   dataRaw_small, resVars, latVars, intLoads, sloLoads,
                                   manMeans, latMeans)

data_mx_small <- mxGenerateData(growthCurveModel_small)

dataRaw_mx_small <- mxData( observed=data_mx_small, type="raw" )
growthCurveModel_small <- mxModel("Linear Growth Curve Model Path Specification", 
                                type="RAM",
                                manifestVars=str_c("t", 1:4),
                                latentVars=c("intercept","slope"),
                                dataRaw_mx_small, 
                                resVars, latVars, intLoads, sloLoads,
                                manMeans, latMeans)

growthCurveFit_small <- mxRun(growthCurveModel_small)
microbenchmark(mxRun(growthCurveModel_small), times = 1)
sum_small = summary(growthCurveFit_small)

# big model ---------------------------------------------------------------
data_big <- data.frame(load_t1 = rep(1, 500))

for(i in 1:15){
  name <- str_c("load_t", i)
  data_big[[name]] <- i
}

data_big %<>% mutate(across(contains("load"), function(x)x+rnorm(500, 0, 0.5)))

#dataRaw <- mxData( observed=Demo.growth, type="raw" )
# residual variances
resVars <- mxPath( from=str_c("t", 1:15), arrows=2,
                   free=TRUE, values = rep(1, 15))
# latent variances and covariance
latVars <- mxPath( from=c("intercept","slope"), arrows=2, connect="unique.pairs",
                   free=TRUE, values=c(1,1,1), labels=c("vari","cov","vars") )
# intercept loadings
intLoads <- mxPath( from="intercept", to=str_c("t", 1:15), arrows=1,
                    free=FALSE, values=rep(1,15) )
# slope loadings
sloLoads <- mxPath( from="slope", to=str_c("t", 1:15), arrows=1,
                    free=FALSE, #values=c(0,1,2,3), 
                    labels = str_c("data.load_t", 1:15))
# manifest means
manMeans <- mxPath( from="one", to=str_c("t", 1:15), arrows=1,
                    free=FALSE, values=rep(0,15) )
# latent means
latMeans <- mxPath( from="one", to=c("intercept", "slope"), arrows=1,
                    free=TRUE, values=c(1,0.5), labels=c("meani","means") )

dataRaw_big <- mxData( observed=data_big, type="raw" )

growthCurveModel_big <- mxModel("Linear Growth Curve Model Path Specification", 
                                type="RAM",
                                manifestVars=str_c("t",1:15),
                                latentVars=c("intercept","slope"),
                                dataRaw_big, 
                                resVars, latVars, intLoads, sloLoads,
                                manMeans, latMeans)

data_mx_big <- mxGenerateData(growthCurveModel_big)

dataRaw_mx_big <- mxData( observed=data_mx_big, type="raw" )
growthCurveModel_big <- mxModel("Linear Growth Curve Model Path Specification", 
                                type="RAM",
                                manifestVars=str_c("t", 1:15),
                                latentVars=c("intercept","slope"),
                                dataRaw_mx_big, 
                                resVars, latVars, intLoads, sloLoads,
                                manMeans, latMeans)

growthCurveFit_big <- mxRun(growthCurveModel_big)
microbenchmark(mxRun(growthCurveModel_big), times = 1)
sum_big = summary(growthCurveFit_big)


# huge model --------------------------------------------------------------

ntime = 30
nobs = 500

data_huge <- data.frame(load_t1 = rep(1, nobs))

for(i in 1:30){
  name <- str_c("load_t", i)
  data_huge[[name]] <- i
}

data_huge %<>% mutate(across(contains("load"), function(x)x+rnorm(nobs, 0, 0.5)))

#dataRaw <- mxData( observed=Demo.growth, type="raw" )
# residual variances
resVars <- mxPath( from=str_c("t", 1:ntime), arrows=2,
                   free=TRUE, values = rep(1, ntime))
# latent variances and covariance
latVars <- mxPath( from=c("intercept","slope"), arrows=2, connect="unique.pairs",
                   free=TRUE, values=c(1,1,1), labels=c("vari","cov","vars") )
# intercept loadings
intLoads <- mxPath( from="intercept", to=str_c("t", 1:ntime), arrows=1,
                    free=FALSE, values=rep(1,15) )
# slope loadings
sloLoads <- mxPath( from="slope", to=str_c("t", 1:ntime), arrows=1,
                    free=FALSE, #values=c(0,1,2,3), 
                    labels = str_c("data.load_t", 1:ntime))
# manifest means
manMeans <- mxPath( from="one", to=str_c("t", 1:ntime), arrows=1,
                    free=FALSE, values=rep(0,ntime) )
# latent means
latMeans <- mxPath( from="one", to=c("intercept", "slope"), arrows=1,
                    free=TRUE, values=c(1,1), labels=c("meani","means") )

dataRaw_huge <- mxData( observed=select(data_huge, contains("load")), type="raw" )

growthCurveModel_huge <- mxModel("Linear Growth Curve Model Path Specification", 
                                type="RAM",
                                manifestVars=str_c("t",1:ntime),
                                latentVars=c("intercept","slope"),
                                dataRaw_huge, 
                                resVars, latVars, intLoads, sloLoads,
                                manMeans, latMeans)

data_mx_huge <- mxGenerateData(growthCurveModel_huge)

dataRaw_mx_huge <- mxData( observed=data_mx_huge, type="raw" )
growthCurveModel_huge <- mxModel("Linear Growth Curve Model Path Specification", 
                                type="RAM",
                                manifestVars=str_c("t", 1:ntime),
                                latentVars=c("intercept","slope"),
                                dataRaw_mx_huge, 
                                resVars, latVars, intLoads, sloLoads,
                                manMeans, latMeans)

growthCurveFit_huge <- mxRun(growthCurveModel_huge)
microbenchmark(mxRun(growthCurveModel_huge), times = 1)
sum_huge = summary(growthCurveFit_huge)


# write -------------------------------------------------------------------

data_small <- select(data_mx_small, starts_with("t"))
data_big <- select(data_mx_big, starts_with("t"))
data_huge <- select(data_mx_huge, starts_with("t"))

data_def_small <- select(data_mx_small, starts_with("load"))
data_def_big <- select(data_mx_big, starts_with("load"))
data_def_huge <- select(data_mx_huge, starts_with("load"))

write_feather(data_small, 
              str_c("test/comparisons/data_unique_small.feather"))
write_feather(data_big, 
              str_c("test/comparisons/data_unique_big.feather"))
write_feather(data_huge, 
              str_c("test/comparisons/data_unique_huge.feather"))

write_feather(data_def_small, 
              str_c("test/comparisons/data_def_unique_small.feather"))
write_feather(data_def_big, 
              str_c("test/comparisons/data_def_unique_big.feather"))
write_feather(data_def_huge, 
              str_c("test/comparisons/data_def_unique_huge.feather"))

pars_unique_small <- sum_small$parameters %>% 
  select(row, col, Estimate, Std.Error)
pars_unique_big <- sum_big$parameters %>% 
  select(row, col, Estimate, Std.Error)
pars_unique_huge <- sum_huge$parameters %>% 
  select(row, col, Estimate, Std.Error)

write_feather(pars_unique_small, 
              str_c("test/comparisons/pars_unique_small.feather"))
write_feather(pars_unique_big, 
              str_c("test/comparisons/pars_unique_big.feather"))
write_feather(pars_unique_huge, 
              str_c("test/comparisons/pars_unique_huge.feather"))


