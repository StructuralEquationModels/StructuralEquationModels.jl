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

Demo.growth %<>% mutate(
  load_t1 = rep(0, 400),
  load_t2 = c(rep(0.5, 200), rep(1.5, 200)),
  load_t3 = c(rep(1.5, 200), rep(2.5, 200)),
  load_t4 = c(rep(2.5, 200), rep(3.5, 200))
)

dataRaw <- mxData( observed=Demo.growth, type="raw" )
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
growthCurveModel <- mxModel("Linear Growth Curve Model Path Specification", 
                            type="RAM",
                            manifestVars=c("t1","t2","t3","t4"),
                            latentVars=c("intercept","slope"),
                            dataRaw, resVars, latVars, intLoads, sloLoads,
                            manMeans, latMeans)

growthCurveFit <- mxRun(growthCurveModel)
#microbenchmark(mxRun(growthCurveModel))
sum = summary(growthCurveFit)

def_pars <- sum$parameters %>% select(row, col, Estimate, Std.Error)



# with unique loadings and missings ---------------------------------------

Demo.growth_missing <- mutate(Demo.growth, 
                              across(starts_with("t"), ~induce_missing(., 0.3)))

Demo.growth_missing_unique <-  mutate(Demo.growth_missing,
                                      load_t1 = load_t1 + rnorm(400, 0, 0.5),
                                      load_t2 = load_t2 + rnorm(400, 0, 0.5),
                                      load_t3 = load_t3 + rnorm(400, 0, 0.5),
                                      load_t4 = load_t4 + rnorm(400, 0, 0.5)
)

Demo.growth_unique <- mutate(Demo.growth, 
                             load_t1 = Demo.growth_missing_unique$load_t1,
                             load_t2 = Demo.growth_missing_unique$load_t2,
                             load_t3 = Demo.growth_missing_unique$load_t3,
                             load_t4 = Demo.growth_missing_unique$load_t4)

dataRaw_unique <- mxData( observed=Demo.growth_unique, type="raw" )
dataRaw_missing <- mxData( observed=Demo.growth_missing, type="raw" )
dataRaw_missing_unique <- mxData( observed=Demo.growth_missing_unique, type="raw" )

growthCurveModel_unique <- mxModel("Linear Growth Curve Model Path Specification", 
                            type="RAM",
                            manifestVars=c("t1","t2","t3","t4"),
                            latentVars=c("intercept","slope"),
                            dataRaw_unique, resVars, latVars, intLoads, sloLoads,
                            manMeans, latMeans)

growthCurveFit_unique <- mxRun(growthCurveModel_unique)
#microbenchmark(mxRun(growthCurveModel))
sum = summary(growthCurveFit_unique)

growthCurveModel_missing <- mxModel("Linear Growth Curve Model Path Specification", 
                                    type="RAM",
                                    manifestVars=c("t1","t2","t3","t4"),
                                    latentVars=c("intercept","slope"),
                                    dataRaw_missing, resVars, latVars, intLoads, sloLoads,
                                    manMeans, latMeans)
growthCurveFit_missing <- mxRun(growthCurveModel_missing)
sum_missing = summary(growthCurveFit_missing)
def_pars_missing <- sum_missing$parameters %>% select(row, col, Estimate, Std.Error)

growthCurveModel_missing_unique <- mxModel("Linear Growth Curve Model Path Specification", 
                                           type="RAM",
                                           manifestVars=c("t1","t2","t3","t4"),
                                           latentVars=c("intercept","slope"),
                                           dataRaw_missing_unique, resVars, latVars, intLoads, sloLoads,
                                           manMeans, latMeans)
growthCurveFit_missing_unique <- mxRun(growthCurveModel_missing_unique)
sum_missing_unique = summary(growthCurveFit_missing_unique)
def_pars_missing_unique <- sum_missing_unique$parameters %>% select(row, col, Estimate, Std.Error)


data_growth <- select(Demo.growth, t1, t2, t3, t4)
data_growth_miss_30 <- select(Demo.growth_missing, t1, t2, t3, t4)

write_feather(data_growth, str_c("test/comparisons/growth_dat.feather"))
write_feather(data_growth_miss_30, str_c("test/comparisons/growth_dat_miss30.feather"))

data_definition <- select(Demo.growth, starts_with("load"))
data_definition_unique <- select(Demo.growth_missing_unique, starts_with("load"))

write_feather(data_definition, str_c("test/comparisons/definition_dat.feather"))
write_feather(data_definition_unique, str_c("test/comparisons/definition_dat_unique.feather"))
write_feather(def_pars, str_c("test/comparisons/definition_par.feather"))
write_feather(def_pars_missing, str_c("test/comparisons/definition_par_missing.feather"))
write_feather(def_pars_missing_unique, str_c("test/comparisons/definition_par_missing_unique.feather"))


