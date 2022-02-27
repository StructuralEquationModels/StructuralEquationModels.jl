pacman::p_load(dplyr, lavaan)

set.seed(62372)

model <- '  visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9 '

data <- HolzingerSwineford1939

fit_ml <- cfa(model, 
           data, 
           group = "school",
           group.equal = c("loadings"),
           likelihood = "wishart")

fit_ls <- cfa(model, 
           data, 
           group = "school",
           group.equal = c("loadings"),
           estimator = "GLS")

par_ml <- select(parTable(fit_ml), lhs, op, rhs, est, start, free)
par_ls <- select(parTable(fit_ls), lhs, op, rhs, est, start, free)

write.csv(par_ml, "test/examples/data/par_multigroup_ml.csv")
write.csv(par_ls, "test/examples/data/par_multigroup_ls.csv")
write.csv(data, "test/examples/data/data_multigroup.csv")