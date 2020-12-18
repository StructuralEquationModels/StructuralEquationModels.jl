pacman::p_load(here, feather, tidyverse, lavaan, microbenchmark, magrittr)

set.seed(123)

#setwd(r"(C:\Users\maxim\.julia\dev\sem)")

#----lavaan----
models <- c(one_fact = "f1 =~ x1 + x2 + x3",
            three_path =
            "# measurement model
            ind60 =~ x1 + x2 + x3
            dem60 =~ y1 + y2 + y3 + y4
            dem65 =~ y5 + y6 + y7 + y8
            # regressions
            dem60 ~ ind60
            dem65 ~ ind60 + dem60
            # residual correlations
            y1 ~~ y5
            y2 ~~ y4 + y6
            y3 ~~ y7
            y4 ~~ y8
            y6 ~~ y8",
            three_mean =
            "# three-factor model
            visual =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            # intercepts with fixed values
            x1 + x2 + x3 + x4 ~ 0.5*1",
            three_path_2 =
            "# measurement model
            ind60 =~ x1 + x2 + x3
            dem60 =~ y1 + y2 + y3 + y4
            dem65 =~ y5 + y6 + y7 + y8
            # regressions
            dem60 ~ ind60
            dem65 ~ ind60 + dem60
            # residual correlations
            y1 ~~ y5
            y2 ~~ y4 + y6
            y3 ~~ y7
            y4 ~~ y8
            y6 ~~ y8",
            three_path_loadeq =
            "# measurement model
            ind60 =~ x1 + x2 + x3
            dem60 =~ y1 + y2 + y3 + y4
            dem65 =~ y5 + y6 + y7 + y8
            # regressions
            dem60 ~ ind60
            dem65 ~ ind60 + dem60
            # residual correlations
            y1 ~~ y5
            y2 ~~ y4 + y6
            y3 ~~ y7
            y4 ~~ y8
            y6 ~~ y8",
            three_path_mean =
            "# measurement model
            ind60 =~ x1 + x2 + x3
            dem60 =~ y1 + y2 + y3 + y4
            dem65 =~ y5 + y6 + y7 + y8
            # regressions
            dem60 ~ ind60
            dem65 ~ ind60 + dem60
            # residual correlations
            y1 ~~ y5
            y2 ~~ y4 + y6
            y3 ~~ y7
            y4 ~~ y8
            y6 ~~ y8
            #means
            x1 + x2 + x3 ~ a*1
            y6 ~ 3*1"
            )

datas <-  list(one_fact = HolzingerSwineford1939,
             three_path = PoliticalDemocracy,
             three_mean = HolzingerSwineford1939,
             three_path_2 = PoliticalDemocracy,
             three_path_loadeq = bind_rows(PoliticalDemocracy, PoliticalDemocracy),
             three_path_mean = PoliticalDemocracy)

datas[[4]]$group = c(rep("1", 25), rep("2", 25), rep("3", 25))
datas[[5]]$group = c(rep("1", 40), rep("2", 40), rep("3", 70))

fits <- map2(models, datas, ~cfa(.x, .y, meanstructure = T))

all_equal = c("loadings", "residuals", "residual.covariances", "lv.variances",
              "lv.covariances", "regressions")

fits[[4]] <- cfa(models[[4]], datas[[4]], group = "group", group.equal = all_equal,
                 meanstructure = FALSE)#c("loadings"))
fits[[5]] <- cfa(models[[5]], datas[[5]], group = "group", group.equal = c("loadings",
                                                                           "regressions"),
                 meanstructure = FALSE)


# FIML --------------------------------------------------------------------
 
induce_missing <- function(v, p){
  miss <- sample(c(0,1), length(v), replace = TRUE, prob = c(1-p, p))
  ifelse(miss, NA, v)
}

three_path_dat_miss20 <- mutate(PoliticalDemocracy, 
                                across(everything(), ~induce_missing(., 0.2)))
three_path_dat_miss30 <- mutate(PoliticalDemocracy, 
                                across(everything(), ~induce_missing(., 0.3)))
three_path_dat_miss50 <- mutate(PoliticalDemocracy, 
                             across(everything(), ~induce_missing(., 0.5)))

datas_miss <- list(
  dat_miss20 = three_path_dat_miss20, 
  dat_miss30 = three_path_dat_miss30, 
  dat_miss50 = three_path_dat_miss50)
fits_miss <- map(datas_miss, ~cfa(models[[2]], data = .x, missing = "FIML"))
fits_miss_mean <- 
  map(datas_miss, ~cfa(models[[6]], data = .x, missing = "FIML"))


# write do disk -----------------------------------------------------------

#rams <- map(fits, RAMpath::lavaan2ram) %>% map(~.[c("A", "S")])

get_testpars <- function(fit) {
  select(parameterEstimates(fit), lhs, op, rhs, est, se, p = pvalue, z)
}

#write_feather(datas[[5]], "test/comparisons/testdat.feather")

pars <- map(fits, get_testpars)
pars_miss <- map(fits_miss, get_testpars)
pars_miss_mean <- map(fits_miss_mean, get_testpars)

data_subsets <- map(fits, lavNames, "ov") %>%
  map2(datas, ~select(.y, one_of(.x)))

data_subsets_miss <- map(fits_miss, lavNames, "ov") %>%
  map2(datas_miss, ~select(.y, one_of(.x)))

# write out

imap(data_subsets,
     ~write_feather(.x, str_c("test/comparisons/", .y, "_dat.feather")))
imap(pars,
     ~write_feather(.x, str_c("test/comparisons/", .y, "_par.feather")))

imap(data_subsets_miss,
     ~write_feather(.x, str_c("test/comparisons/", .y, "_dat.feather")))
imap(pars_miss,
     ~write_feather(.x, str_c("test/comparisons/", .y, "_par.feather")))
imap(pars_miss_mean,
     ~write_feather(.x, str_c("test/comparisons/", .y, "_par_mean.feather")))


#----benchmarks----

microbenchmark(cfa(models[["three_path_loadeq"]], datas[["three_path_loadeq"]],
                   group = "group", 
                   group.equal = c("loadings")))

microbenchmark(cfa(models[["three_path"]], datas_miss[[1]], missing = "FIML"))

if(FALSE){
  microbenchmark(cfa(models[["one_fact"]], datas[["one_fact"]]),
                 cfa(models[["three_path"]], datas[["three_path"]]),
                 cfa(models[["three_mean"]], datas[["three_mean"]]),
                 cfa(models[["three_path_2"]], datas[["three_path_2"]]))
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


### with unique loadings and missings
Demo.growth_missing <- mutate(Demo.growth, 
       across(starts_with("t"), ~induce_missing(., 0.3)))

Demo.growth_missing_unique <-  mutate(Demo.growth_missing,
  load_t1 = load_t1 + rnorm(400, 0, 0.5),
  load_t2 = load_t2 + rnorm(400, 0, 0.5),
  load_t3 = load_t3 + rnorm(400, 0, 0.5),
  load_t4 = load_t4 + rnorm(400, 0, 0.5)
)

dataRaw_missing <- mxData( observed=Demo.growth_missing, type="raw" )
dataRaw_missing_unique <- mxData( observed=Demo.growth_missing_unique, type="raw" )

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
write_feather(
  select(parameterEstimates(growth_fit), lhs, op, rhs, est, se, p = pvalue, z),
  str_c("test/comparisons/growth_par.feather"))

data_definition <- select(Demo.growth, starts_with("load"))
data_definition_unique <- select(Demo.growth_missing_unique, starts_with("load"))

write_feather(data_definition, str_c("test/comparisons/definition_dat.feather"))
write_feather(data_definition_unique, str_c("test/comparisons/definition_dat_unique.feather"))
write_feather(def_pars, str_c("test/comparisons/definition_par.feather"))
write_feather(def_pars_missing, str_c("test/comparisons/definition_par_missing.feather"))
write_feather(def_pars_missing_unique, str_c("test/comparisons/definition_par_missing_unique.feather"))


# open MX -----------------------------------------------------------------

library(OpenMx)

dataRaw      <- mxData( observed=datas[[2]], type="raw" )
# residual variances
resVars      <- mxPath( from=c("x1", "x2", "x3", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"), arrows=2,
                        free=TRUE, values=c(1,1,1,1,1,1,1,1,1,1,1),
                        labels=c("e1","e2","e3","e4","e5","e6", "e7","e8","e9","e10","e11") )

resVars2      <- mxPath( from=c("y1", "y2", "y2", "y3", "y4", "y6"), 
                        to = c("y5", "y4", "y6", "y7", "y8", "y8"), arrows = 2,
                        free=TRUE, values=rep(0,6),
                        labels=c("ce1","ce2","ce3","ce4","ce5","ce6") )
# latent variances and covariance
latPaths      <- mxPath( from=c("ind60","ind60", "dem60"), to=c("dem65","dem60", "dem65"), arrows=1,
                        free=TRUE, values=c(0,0,0), labels=c("vlat1","vlat2","vlat3") )
latVars      <- mxPath( from=c("ind60","ind60", "dem60"), arrows=2,
                         free=TRUE, values=c(.05,.05,.05), labels=c("lat1","lat2","lat3") )
# factor loadings for x variables
facLoadsX    <- mxPath( from="ind60", to=c("x1","x2","x3"), arrows=1,
                        free=c(F,T,T), values=c(1,1,1), labels=c("l1","l2","l3") )
# factor loadings for y variables
facLoadsY1    <- mxPath( from="dem60", to=c("y1","y2","y3", "y4"), arrows=1,
                        free=c(F,T,T,T), values=c(1,1,1,1), labels=c("l4","l5","l6","l7") )
facLoadsY2   <- mxPath( from="dem65", to=c("y5","y6","y7","y8"), arrows=1,
                        free=c(F,T,T,T), values=c(1,1,1,1), labels=c("l8","l9","l10","l11") )
# means
means        <- mxPath( from="one", to=c("x1","x2","x3","y1","y2","y3",
                                         "y4", "y5", "y6","y7", "y8",
                                         "ind60","dem60", "dem65"),
                        arrows=1,
                        free=c(T,T,T,T,T,T,T,T,T,T,T,F,F), values=c(1,1,1,1,1,1,1,1,1,1,1,0,0,0),
                        labels=c("meanx1","meanx2","meanx3",
                                 "meany1","meany2","meany3",
                                 "meany4","meany5","meany6",
                                 "meany7","meany8",
                                 NA,NA,NA) )

twoFactorModel <- mxModel("Two Factor Model Path Specification", type="RAM",
                          manifestVars=c("x1", "x2", "x3", "y1", "y2", "y3", "y4", "y5", "y6","y7", "y8"),
                          latentVars=c("ind60","dem60", "dem65"),
                          dataRaw, resVars, resVars2, latVars, latPaths, facLoadsX, facLoadsY1, facLoadsY2, means)


oneFactorFit <- mxRun(twoFactorModel)

oneFactorFit$output
summary(oneFactorFit)
