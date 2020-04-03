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
            x1 + x2 + x3 + x4 ~ 0.5*1"
            )

datas <-  list(one_fact = HolzingerSwineford1939,
             three_path = PoliticalDemocracy,
             three_mean = HolzingerSwineford1939)

fits <- map2(models, datas, ~cfa(.x, .y))

rams <- map(fits, RAMpath::lavaan2ram) %>% map(~.[c("A", "S")])

get_testpars <- function(fit) {
  select(parameterEstimates(fit), lhs, op, rhs, est, se, p = pvalue, z)
}

data_subsets <- map(fits, lavNames, "ov") %>%
  map2(datas, ~select(.y, one_of(.x)))

# write out

imap(data_subsets,
     ~write_feather(.x, str_c("test/comparisons/", .y, "_dat.feather")))
imap(pars,
     ~write_feather(.x, str_c("test/comparisons/", .y, "_par.feather")))

#----benchmarks----
if(FALSE){
  microbenchmark(cfa(models[["one_fact"]], datas[["one_fact"]]),
                 cfa(models[["three_path"]], datas[["three_path"]]),
                 cfa(models[["three_mean"]], datas[["three_mean"]]))
}
