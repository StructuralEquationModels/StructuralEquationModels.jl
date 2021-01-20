pacman::p_load(here, feather, tidyverse, magrittr, ggplot2)

#set.seed(123)

setwd(r"(C:\Users\maxim\.julia\dev\sem)")

BFGS_results = read_csv("test/comparisons/benchmark_big.csv")

BFGS_results %<>% mutate(is_finite = !is.infinite(minimum))

BFGS_results %>% ggplot() + geom_point(aes(x = line_search, y = is_finite))

BFGS_results %>% group_by(line_search) %>% summarise(n_inf = sum(!is_finite))
BFGS_results %>% group_by(step_length) %>% summarise(n_inf = sum(!is_finite))
BFGS_results %>% group_by(algo) %>% summarise(n_inf = sum(!is_finite))

BFGS_results %>% group_by(line_search) %>% summarise(n_inf = sum(!truepars), n = n())
BFGS_results %>% group_by(step_length) %>% summarise(n_inf = sum(!truepars), n = n())
BFGS_results %>% group_by(algo) %>% summarise(n_inf = sum(!truepars))

BFGS_results %>% ggplot() + 
  geom_point(aes(x = line_search,
                 y = interaction(step_length, algo, m),
                 color = truepars,
                 shape = algo),
             #position = position_jitter(width = 1),
             alpha = 0.7) +
  theme_minimal()

BFGS_results %<>% filter(truepars)

View(BFGS_results %>% arrange(line_search, step_length))

BFGS_results %>% ggplot() + geom_point(aes(x = algo, y = time, 
                                           color = line_search), alpha = .5)

BFGS_results %>% ggplot() + geom_point(aes(x = step_length, y = time))

BFGS_results %>% ggplot() + geom_point(aes(x = line_search, y = truepars))

BFGS_results %>% ggplot() + geom_point(aes(x = line_search, y = minimum))

BFGS_results %>% ggplot() + 
  geom_point(aes(x = step_length, y = time, color = factor(m),
                group = interaction(line_search, algo, m)), 
             alpha = .5) +
  facet_grid(rows = vars(line_search), cols = vars(algo)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

