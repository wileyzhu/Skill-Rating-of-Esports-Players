library(brms)
library(ggplot2)
library(tidyverse)
library(ggdist)
d <- read.csv(file.choose())
glimpse(d)
d <- d %>%
  select(Game_ID, Player, bo5, final_games, Tourney, State, international, performance_score, own_team, opponent_team) %>%
  mutate(player_id = as.numeric(as.factor(Player)), 
         Game_ID = as.factor(Game_ID),
         own_team = as.factor(own_team),
         opponent_team = as.factor(opponent_team))
players_both <- d %>%
  group_by(player_id) %>%
  summarise(n_playoff = sum(State == "Playoff"),
            n_regular = sum(State == "Regular")) %>%
  filter(n_playoff > 0 & n_regular > 0) %>%
  pull(player_id)

playoff <- d %>%
  filter(State %in% c("Playoffs", "Regular"),
         player_id %in% player_id) %>%
  mutate(pressure = as.integer(State == "Playoffs"))
library(brms)

fit <- brm(
  bf(performance_score ~ pressure + (pressure | player_id)),
  family = zero_one_inflated_beta(),
  data = playoff,
  prior = c(
    prior(normal(0, 0.5), class = "b", coef = "pressure"),
    prior(normal(0, 1), class = "Intercept"),
    prior(exponential(2), class = "sd"),
    prior(lkj(2), class = "cor"),
    prior(exponential(1), class = "phi"),
    prior(beta(2,10), class = "zoi"),  # small inflation
    prior(beta(2,10), class = "coi")  # weakly favor neither 0 nor 1 inside inflation
  ),
  chains = 4, cores = 4,
  iter = 2000, warmup = 1000,
  seed = 123
)
fit1 <- brm(
  bf(performance_score ~ pressure + (pressure | player_id) + (1 | opponent_team)),
  family = zero_one_inflated_beta(),
  data = playoff,
  prior = c(
    prior(normal(0, 0.5), class = "b", coef = "pressure"),
    prior(normal(0, 1), class = "Intercept"),
    prior(exponential(2), class = "sd"),
    prior(lkj(2), class = "cor"),
    prior(exponential(1), class = "phi"),
    prior(beta(2,10), class = "zoi"),  # small inflation
    prior(beta(2,10), class = "coi")  # weakly favor neither 0 nor 1 inside inflation
  ),
  chains = 4, cores = 4,
  iter = 2000, warmup = 1000,
  seed = 123
)

# Player-level deviations (posterior draws)
post <- as_draws_df(fit)

# Get population pressure effect draws
b_press <- post$`b_pressure`

# Extract random slopes by player as draws:
# Column names like r_player_id[PLAYER,pressure]
slopem <- post %>%
  select(starts_with("r_player_id[") & ends_with(",pressure]")) %>%
  as_tibble()

# Map column names to player ids
player_names <- gsub("^r_player_id\\[|,pressure\\]$", "", colnames(slopem))

# Compute total slopes per player (logit scale)
tot_slopes <- slopem %>%
  mutate(draw = row_number()) %>%
  bind_cols(tibble(b_press = b_press)) %>%
  pivot_longer(cols = starts_with("r_player_id["), names_to = "player", values_to = "u1") %>%
  mutate(player = gsub("^r_player_id\\[|,pressure\\]$", "", player),
         beta_player = b_press + u1)

# Summarise: median & CI + Pr(beta_player > 0)
sum_slopes <- tot_slopes %>%
  group_by(player) %>%
  summarise(
    med = median(beta_player),
    lo  = quantile(beta_player, .025),
    hi  = quantile(beta_player, .975),
    pr_pos = mean(beta_player > 0)
  ) %>%
  arrange(med) %>%
  mutate(rank = row_number())

ggplot(sum_slopes, aes(x = med, y = reorder(player, med),
                       xmin = lo, xmax = hi, color = pr_pos)) +
  geom_point(size = 1.5) +
  geom_errorbarh(height = 0) +
  scale_color_viridis_c(name = "Pr(slope>0)") +
  labs(x = "Player-specific pressure effect (logit scale)",
       y = NULL,
       title = "Clutch slopes by player",
       subtitle = "Dots: posterior median; bars: 95% CrI") +
  theme_minimal(base_size = 11) +
  scale_y_discrete(NULL, breaks = NULL)

# Player-level deviations (posterior draws)
post <- as_draws_df(fit1)

# Get population pressure effect draws
b_press <- post$`b_pressure`

# Extract random slopes by player as draws:
# Column names like r_player_id[PLAYER,pressure]
slopem <- post %>%
  select(starts_with("r_player_id[") & ends_with(",pressure]")) %>%
  as_tibble()

# Map column names to player ids
player_names <- gsub("^r_player_id\\[|,pressure\\]$", "", colnames(slopem))

# Compute total slopes per player (logit scale)
tot_slopes <- slopem %>%
  mutate(draw = row_number()) %>%
  bind_cols(tibble(b_press = b_press)) %>%
  pivot_longer(cols = starts_with("r_player_id["), names_to = "player", values_to = "u1") %>%
  mutate(player = gsub("^r_player_id\\[|,pressure\\]$", "", player),
         beta_player = b_press + u1)

# Summarise: median & CI + Pr(beta_player > 0)
sum_slopes <- tot_slopes %>%
  group_by(player) %>%
  summarise(
    med = median(beta_player),
    lo  = quantile(beta_player, .025),
    hi  = quantile(beta_player, .975),
    pr_pos = mean(beta_player > 0)
  ) %>%
  arrange(med) %>%
  mutate(rank = row_number())

ggplot(sum_slopes, aes(x = med, y = reorder(player, med),
                       xmin = lo, xmax = hi, color = pr_pos)) +
  geom_point(size = 1.5) +
  geom_errorbarh(height = 0) +
  scale_color_viridis_c(name = "Pr(slope>0)") +
  labs(x = "Player-specific pressure effect (logit scale)",
       y = NULL,
       title = "Clutch slopes by player",
       subtitle = "Dots: posterior median; bars: 95% CrI") +
  theme_minimal(base_size = 11) +
  scale_y_discrete(NULL, breaks = NULL)

library(dplyr)
library(tidyr)
library(tibble)
library(posterior)

make_sum_slopes <- function(fit) {
  post <- as_draws_df(fit)
  
  # global pressure effect
  b_press <- post$`b_pressure`
  
  # random slopes for players
  slopem <- post %>%
    select(starts_with("r_player_id[") & ends_with(",pressure]")) %>%
    as_tibble()
  
  # player IDs
  player_names <- gsub("^r_player_id\\[|,pressure\\]$", "", colnames(slopem))
  
  # combine into long df
  tot_slopes <- slopem %>%
    mutate(draw = row_number()) %>%
    bind_cols(tibble(b_press = b_press)) %>%
    pivot_longer(cols = starts_with("r_player_id["), 
                 names_to = "player", values_to = "u1") %>%
    mutate(player = gsub("^r_player_id\\[|,pressure\\]$", "", player),
           beta_player = b_press + u1)
  
  # summarise per player
  sum_slopes <- tot_slopes %>%
    group_by(player) %>%
    summarise(
      med = median(beta_player, na.rm = TRUE),
      lo  = quantile(beta_player, 0.025, na.rm = TRUE),
      hi  = quantile(beta_player, 0.975, na.rm = TRUE),
      pr_pos = mean(beta_player > 0, na.rm = TRUE)
    ) %>%
    arrange(med) %>%
    mutate(rank = row_number(),
           label = ifelse(rank %% 50 == 0, player, NA))  # label every 50th player for clarity
  
  return(sum_slopes)
}

# Create for both models
sum_slopes1 <- make_sum_slopes(fit)   # without opponent adjustment
sum_slopes2 <- make_sum_slopes(fit1)   # with opponent adjustment

library(ggplot2)
library(dplyr)
library(patchwork)


# ---- Without opponent adjustment ----
plot_slopes <- function(sum_slopes, fit, title, cmap, lims) {
  ggplot(sum_slopes, aes(x = med, y = reorder(player, med),
                         xmin = lo, xmax = hi, color = pr_pos)) +
    geom_errorbarh(height = 0, size = 0.3) +
    geom_point(size = 1.0, color = "black") +
    geom_vline(xintercept = fixef(fit)["pressure","Estimate"],
               linetype = "dashed", color = "black") +
    geom_vline(xintercept = 0, linetype = "dotted", color = "grey50") +
    scale_color_viridis_c(name = "Pr(slope>0)", option = cmap, limits = lims) +
    labs(x = "Player-specific pressure effect (logit scale)",
         y = NULL,
         title = title,
         subtitle = "Dots = median; bars = 95% CrI; dashed = population effect") +
    theme_minimal(base_size = 11) +
    theme(legend.position = "right") +
    scale_y_discrete(breaks = NULL)
}

# Left: without opponent adjustment (small probs)
p1 <- plot_slopes(sum_slopes1, fit, "Without opponent adjustment",
                  cmap = "magma", lims = c(0,0.1))

# Right: with opponent adjustment (wide spread)
p2 <- plot_slopes(sum_slopes2, fit1, "With opponent adjustment",
                  cmap = "viridis", lims = c(0,1))

p1 + p2

# keep only players who have both international=1 and international=0 records
players_both1 <- d %>%
  group_by(player_id) %>%
  summarise(n_intl = sum(international == 1),
            n_domestic = sum(international == 0)) %>%
  filter(n_intl > 0 & n_domestic > 0) %>%
  pull(player_id)

inter <- d %>%
  filter(player_id %in% players_both1) %>%
  mutate(pressure = as.integer(international))
fit2 <- brm(
  bf(performance_score ~ pressure + (pressure | player_id)),
  family = zero_one_inflated_beta(),
  data = inter,
  prior = c(
    prior(normal(0, 0.5), class = "b", coef = "pressure"),
    prior(normal(0, 1), class = "Intercept"),
    prior(exponential(2), class = "sd"),
    prior(lkj(2), class = "cor"),
    prior(exponential(1), class = "phi"),
    prior(beta(2,10), class = "zoi"),  # small inflation
    prior(beta(2,10), class = "coi")  # weakly favor neither 0 nor 1 inside inflation
  ),
  chains = 4, cores = 4,
  iter = 2000, warmup = 1000,
  seed = 123
)
fit3 <- brm(
  bf(performance_score ~ pressure + (pressure | player_id) + (1 | opponent_team)),
  family = zero_one_inflated_beta(),
  data = inter,
  prior = c(
    prior(normal(0, 0.5), class = "b", coef = "pressure"),
    prior(normal(0, 1), class = "Intercept"),
    prior(exponential(2), class = "sd"),
    prior(lkj(2), class = "cor"),
    prior(exponential(1), class = "phi"),
    prior(beta(2,10), class = "zoi"),  # small inflation
    prior(beta(2,10), class = "coi")  # weakly favor neither 0 nor 1 inside inflation
  ),
  chains = 4, cores = 4,
  iter = 2000, warmup = 1000,
  seed = 123
)

sum_slopes3 <- make_sum_slopes(fit2)   # without opponent adjustment
sum_slopes4 <- make_sum_slopes(fit3)   # with opponent adjustment

p3 <- plot_slopes(sum_slopes3, fit2, "Without opponent adjustment",
                  cmap = "magma", lims = c(0,1))

# Right: with opponent adjustment (wide spread)
p4 <- plot_slopes(sum_slopes4, fit3, "With opponent adjustment",
                  cmap = "viridis", lims = c(0,1))

bo5 <- d %>%
  filter(bo5 == "True") %>%
  mutate(pressure = ifelse(final_games == "True", 1, 0))
fit4 <- brm(
  bf(performance_score ~ pressure + (pressure | player_id)),
  family = zero_one_inflated_beta(),
  data = bo5,
  prior = c(
    prior(normal(0, 0.5), class = "b", coef = "pressure"),
    prior(normal(0, 1), class = "Intercept"),
    prior(exponential(2), class = "sd"),
    prior(lkj(2), class = "cor"),
    prior(exponential(1), class = "phi"),
    prior(beta(2,10), class = "zoi"),  # small inflation
    prior(beta(2,10), class = "coi")  # weakly favor neither 0 nor 1 inside inflation
  ),
  chains = 4, cores = 4,
  iter = 2000, warmup = 1000,
  seed = 123
)

fit5 <- brm(
  bf(performance_score ~ pressure + (pressure | player_id) + (1 | opponent_team)),
  family = zero_one_inflated_beta(),
  data = bo5,
  prior = c(
    prior(normal(0, 0.5), class = "b", coef = "pressure"),
    prior(normal(0, 1), class = "Intercept"),
    prior(exponential(2), class = "sd"),
    prior(lkj(2), class = "cor"),
    prior(exponential(1), class = "phi"),
    prior(beta(2,10), class = "zoi"),  # small inflation
    prior(beta(2,10), class = "coi")  # weakly favor neither 0 nor 1 inside inflation
  ),
  chains = 4, cores = 4,
  iter = 2000, warmup = 1000,
  seed = 123
)
sum_slopes5 <- make_sum_slopes(fit4)   # without opponent adjustment
sum_slopes6 <- make_sum_slopes(fit5)   # with opponent adjustment
p5 <- plot_slopes(sum_slopes5, fit4, "Without opponent adjustment",
                  cmap = "magma", lims = c(0,0.7))
# Right: with opponent adjustment (wide spread)
p6 <- plot_slopes(sum_slopes6, fit5, "With opponent adjustment",
                  cmap = "viridis", lims = c(0,1))
p5 + p6

p5

fit <- add_criterion(fit, "loo")
fit1 <- add_criterion(fit1, "loo")

fit2 <- add_criterion(fit2, "loo")
fit3 <- add_criterion(fit3, "loo")

library(modelsummary)

library(modelsummary)

# Example: assuming you have lists of models
fits <- list(fit1 = fit, fit2 = fit2, fit3 = fit3, fit4 = fit4, fit5 = fit5)
fits_adj <- list(fit1a = fit1, fit2a = fit2a, fit3a = fit3a, fit4a = fit4a, fit5a = fit5a)

# Loop through indices 1:5
for (i in 1:5) {
  file_name <- paste0("brms_models", i, ".docx")
  
  msummary(
    list("Baseline" = fits[[i]], "Opponent-adjusted" = fits_adj[[i]]),
    statistic = "[{conf.low}, {conf.high}]",
    output = file_name
  )
}