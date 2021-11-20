library(tidyverse)
results <- readr::read_rds("results.rds")
benchmark_omx <- readr::read_rds("results/benchmarks_omx.rds")

benchmark_omx <- mutate(
  select(results, -c(data, model, model_lavaan, model_omx, start, ends_with("lav"))),
  benchmark_omx = benchmark_omx
)
benchmark_omx <- unnest(benchmark_omx, benchmark_omx)
benchmark_omx <- mutate(benchmark_omx, result = as.double(result, unit = "secs"))

results %>% 
  filter(Estimator == "ML") %>% 
  select(starts_with("mean_time"), meanstructure, n_factors, n_items) %>% 
  pivot_longer(starts_with("mean_time"), names_to = "package", values_to = "mean_time") %>% 
  ggplot(aes(x = n_factors*n_items, y = mean_time, color = package)) +
  geom_point() + theme_minimal() + geom_line(aes(linetype = as.factor(meanstructure))) +
  scale_x_log10() + scale_y_log10()

results %>% 
  filter(Estimator == "ML") %>% 
  select(starts_with("mean_time"), meanstructure, n_factors, n_items) %>% 
  mutate(mean_time_lav = mean_time_lav + 0.001) %>% 
  pivot_longer(starts_with("mean_time"), names_to = "package", values_to = "mean_time") %>% 
  ggplot(aes(x = n_factors*n_items, y = mean_time, color = package)) +
  geom_point() + theme_minimal() + geom_line(aes(linetype = as.factor(meanstructure))) +
  scale_x_log10() + scale_y_log10()

results %>% 
  filter(Estimator == "ML") %>% 
  select(starts_with("sd_time"), meanstructure, n_factors, n_items) %>% 
  #mutate(median_time_lav = median_time_lav + 0.001) %>% 
  pivot_longer(starts_with("sd_time"), names_to = "package", values_to = "mean_time") %>% 
  ggplot(aes(x = n_factors*n_items, y = mean_time, color = package)) +
  geom_point() + theme_minimal() + geom_line(aes(linetype = as.factor(meanstructure)))

benchmark_omx %>% 
  filter(Estimator == "ML") %>%
  ggplot(aes(x = n_factors*n_items, y = result, color = meanstructure)) +
  geom_point() + theme_minimal() + 
  scale_x_log10() + scale_y_log10()

lm_data <- filter(benchmark_omx, meanstructure == 0) %>% 
  mutate(n_obs = log10(n_factors*n_items), result = log10(result))

runtime <- lm(result ~ n_obs, lm_data)

lm_data <- filter(results, meanstructure == 0, Estimator == "ML") %>% 
  mutate(n_obs = log10(n_factors*n_items), result = log10(mean_time_lav + 0.001))

runtime <- lm(result ~ n_obs, lm_data)

lm_data %>% ggplot(aes(x = n_obs, y = result)) + geom_point() +
  geom_smooth(method = "lm")
