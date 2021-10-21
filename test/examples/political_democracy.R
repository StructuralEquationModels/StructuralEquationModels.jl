pacman::p_load(dplyr, lavaan)

set.seed(123)

#----lavaan----
model <-    "# measurement model
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

data <- PoliticalDemocracy

data <- select(data, starts_with("x"), starts_with("y"))

fit_ml <- cfa(model, data, likelihood = "wishart", do.fit = TRUE)
fit_ls <- cfa(model, data, estimator = "GLS", do.fit = TRUE)

par_ml <- select(parTable(fit_ml), lhs, op, rhs, est, start)
par_ls <- select(parTable(fit_ls), lhs, op, rhs, est, start)

write.csv(par_ml, "test/examples/data/par_dem_ml.arrow")
write.csv(par_ls, "test/examples/data/par_dem_ls.arrow")
write.csv(data, "test/examples/data/data_dem.arrow")