pacman::p_load(here, feather, tidyverse, magrittr, ggplot)

#set.seed(123)

setwd(r"(C:\Users\maxim\.julia\dev\sem)")

BFGS_results = read_csv("test/comparisons/BFGS_small.csv")

BFGS_results %<>% mutate(is_finite = !is.infinite(minimum))

BFGS_results %>% ggplot() + geom_point(aes(x = line_search, y = is_finite))

BFGS_results %>% group_by(line_search) %>% summarise(n_inf = sum(!is_finite))
BFGS_results %>% group_by(step_length) %>% summarise(n_inf = sum(!is_finite))

BFGS_results %>% group_by(line_search) %>% summarise(n_inf = sum(!truepars))
BFGS_results %>% group_by(step_length) %>% summarise(n_inf = sum(!truepars))

BFGS_results %<>% filter(truepars)

View(BFGS_results %>% arrange(line_search, step_length))

BFGS_results %>% ggplot() + geom_point(aes(x = line_search, y = time, 
                                           color = step_length))

BFGS_results %>% ggplot() + geom_point(aes(x = step_length, y = time))

BFGS_results %>% ggplot() + geom_point(aes(x = line_search, y = truepars))

BFGS_results %>% ggplot() + geom_point(aes(x = line_search, y = minimum))
