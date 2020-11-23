pacman::p_load(here, feather, tidyverse, lavaan, microbenchmark)

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
            y6 ~~ y8"
            )

datas <-  list(one_fact = HolzingerSwineford1939,
             three_path = PoliticalDemocracy,
             three_mean = HolzingerSwineford1939,
             three_path_2 = PoliticalDemocracy,
             three_path_loadeq = bind_rows(PoliticalDemocracy, PoliticalDemocracy))

datas[[4]]$group = c(rep("1", 25), rep("2", 25), rep("3", 25))
datas[[5]]$group = c(rep("1", 40), rep("2", 40), rep("3", 70))

fits <- map2(models, datas, ~cfa(.x, .y, meanstructure = T))

all_equal = c("loadings", "residuals", "residual.covariances", "lv.variances",
              "lv.covariances", "regressions")

fits[[4]] <- cfa(models[[4]], datas[[4]], group = "group", group.equal = all_equal,
                 meanstructure = FALSE)#c("loadings"))
fits[[5]] <- cfa(models[[5]], datas[[5]], group = "group", group.equal = c("loadings"),
                 meanstructure = FALSE)

rams <- map(fits, RAMpath::lavaan2ram) %>% map(~.[c("A", "S")])

get_testpars <- function(fit) {
  select(parameterEstimates(fit), lhs, op, rhs, est, se, p = pvalue, z)
}

#write_feather(datas[[5]], "test/comparisons/testdat.feather")

pars <- map(fits, get_testpars)

data_subsets <- map(fits, lavNames, "ov") %>%
  map2(datas, ~select(.y, one_of(.x)))

# write out

imap(data_subsets,
     ~write_feather(.x, str_c("test/comparisons/", .y, "_dat.feather")))
imap(pars,
     ~write_feather(.x, str_c("test/comparisons/", .y, "_par.feather")))

microbenchmark(cfa(models[["three_path_loadeq"]], datas[["three_path_loadeq"]],
                   group = "group", 
                   group.equal = c("loadings")))

#----benchmarks----
if(FALSE){
  microbenchmark(cfa(models[["one_fact"]], datas[["one_fact"]]),
                 cfa(models[["three_path"]], datas[["three_path"]]),
                 cfa(models[["three_mean"]], datas[["three_mean"]]),
                 cfa(models[["three_path_2"]], datas[["three_path_2"]]))
}


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
