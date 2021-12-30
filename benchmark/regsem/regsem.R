library(regsem)
# put variables on same scale for regsem
HS <- data.frame(scale(HolzingerSwineford1939[,7:15]))
# equivalent to
mod <- '
f =~ 1*x1 + l1*x2 + l2*x3 + l3*x4 + l4*x5 + l5*x6 + l6*x7 + l7*x8 + l8*x9
'
outt = cfa(mod,HS)
# increase to > 25
cv.out <- cv_regsem(outt, type="ridge", pars_pen=c("l1","l2","l6","l7","l8"),
n.lambda=50,jump=0.001)
summary(cv.out)
plot(cv.out, show.minimum="BIC")

write.csv(HS, "benchmark/regsem/data.csv")

#cv.out = cv_regsem(outt, type="ridge", pars_pen=c("l1","l2","l6","l7","l8"),
#n.lambda=1,jump=0.001, lambda.start = 0.02)

library(microbenchmark)

bm = microbenchmark(
    cv_regsem(outt, type="ridge", pars_pen=c("l1","l2","l6","l7","l8"),
    n.lambda=50,jump=0.001),
    times = 10,
    unit = "ms")

summary(bm)

cv.out = cv_regsem(outt, type="lasso", pars_pen=c("l1","l2","l6","l7","l8"),
    n.lambda=21,jump=0.01, lambda.start = 0.0)

plot(cv.out)

png("/home/maximilian/Downloads/cv_out.png")

dev.off()

system.time(
    cv_regsem(outt, type="lasso", pars_pen=c("l1","l2","l6","l7","l8"),
    n.lambda=21,jump=0.01, lambda.start = 0.0))

cv.out = cv_regsem(outt, type="lasso", pars_pen=c("l1","l2","l6","l7","l8"),
    n.lambda=10,jump=0.1, lambda.start = 0.0)
