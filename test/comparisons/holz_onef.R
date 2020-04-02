pacman::p_load(lavaan, feather, here)

data(HolzingerSwineford1939)
dat <- HolzingerSwineford1939[7:9]

model <- "f1 =~ x1 + x2 + x3"

testfit <- cfa(model, dat)

testpar <- parameterEstimates(testfit)

testpar <- data.frame(
  lhs = testpar$lhs,
  op = testpar$op,
  rhs = testpar$rhs,
  est = testpar$est,
  se = testpar$se,
  p = testpar$pvalue,
  z = testpar$z
)

write_feather(
  testpar,
  "test/comparisons/holz_onef_par.feather"
  )

write_feather(
  dat,
  "test/comparisons/holz_onef_dat.feather"
  )

manifests <- names(dat)
latents <- c("G")
factorModel <- mxModel("One Factor",
                       mxMatrix("Full", 3, 1, values=1.0,
                                free=c(FALSE, TRUE, TRUE), name="A"),
                       mxMatrix("Symm", 1, 1, values=0.5,
                                free=TRUE, name="L"),
                       mxMatrix("Diag", 3, 3, values=0.5,
                                free=TRUE, name="U"),
                       mxAlgebra(A %*% L %*% t(A) + U, name="R"),
                       mxExpectationNormal(covariance = "R",
                                           dimnames = names(dat)),
                       mxFitFunctionML(),
                       mxData(cov(dat), type="cov", numObs=301))

summary(factorModelFit <- mxRun(factorModel))
