library(dplyr)
library(OpenMx)
library(stringr)

set.seed(2343253)

ntime = 30
nobs = 500

data <- data.frame(load_t1 = rep(1, nobs))

for(i in 1:30){
  name <- str_c("load_t", i)
  data[[name]] <- i
}

data <- mutate(data, across(contains("load"), function(x)x+rnorm(nobs, 0, 0.1)))


# true model to simulate data from ----------------------------------------

# residual variances
resVars <- mxPath( from=str_c("t", 1:ntime), arrows=2,
                   free=TRUE, values = rnorm(ntime, 5, 2))
# latent variances and covariance
latVars <- mxPath( from=c("intercept","slope"), arrows=2, connect="unique.pairs",
                   free=TRUE, values=c(1.8,0.1,0.2), labels=c("vari","cov","vars") )
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
                    free=TRUE, values=c(1.4,0.6), labels=c("meani","means") )


dataRaw <- mxData( observed=select(data, contains("load")), type="raw" )

growthCurveModel_true <- mxModel("Linear Growth Curve Model Path Specification", 
                                      type="RAM",
                                      manifestVars=str_c("t",1:ntime),
                                      latentVars=c("intercept","slope"),
                                      dataRaw, 
                                      resVars, latVars, intLoads, sloLoads,
                                      manMeans, latMeans)

# simulated data
data_mx <- mxGenerateData(growthCurveModel_true)

dataRaw_mx <- mxData(observed=data_mx, type="raw")



# model to fit ------------------------------------------------------------

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

dataRaw_huge <- mxData( observed=data_huge, type="raw" )

growthCurveModel <- mxModel("Linear Growth Curve Model Path Specification", 
                                 type="RAM",
                                 manifestVars=str_c("t", 1:ntime),
                                 latentVars=c("intercept","slope"),
                                 dataRaw_mx, 
                                 resVars, latVars, intLoads, sloLoads,
                                 manMeans, latMeans)

growthCurveFit <- mxRun(growthCurveModel)
summary(growthCurveFit)

omxDetectCores() # returns 8
getOption('mxOptions')$"Number of Threads" # returns 2
mxOption(model= growthCurveModel, key="Number of Threads", 
         value= (omxDetectCores() - 1)) 
#does not change time to fit the model, 
#regardless of the value I pass

growthCurveFit <- mxRun(growthCurveModel)
summary(growthCurveFit)
