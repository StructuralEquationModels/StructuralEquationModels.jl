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
