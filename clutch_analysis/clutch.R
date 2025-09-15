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
  seed = 123,
  file = "fits/fit"
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
  seed = 123,
  file = "fits/fit1"
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
               linetype = "dashed", color = "red", size = 0.8) +
    scale_color_viridis_c(name = "Pr(slope>0)", option = cmap, limits = lims) +
    labs(x = "Player-specific pressure effect (logit scale)",
         y = NULL,
         title = title) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "right") +
    scale_y_discrete(breaks = NULL)
}

# Left: without opponent adjustment (small probs)
p1 <- plot_slopes(sum_slopes1, fit, "Without opponent adjustment",
                  cmap = "magma", lims = c(0,0.3))

# Right: with opponent adjustment (wide spread)
p2 <- plot_slopes(sum_slopes2, fit1, "With opponent adjustment",
                  cmap = "viridis", lims = c(0.4,0.9))

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
  seed = 123,
  file = "fits/fit2"
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
  seed = 123,
  file = "fits/fit3"
)

sum_slopes3 <- make_sum_slopes(fit2)   # without opponent adjustment
sum_slopes4 <- make_sum_slopes(fit3)   # with opponent adjustment

p3 <- plot_slopes(sum_slopes3, fit2, "Without opponent adjustment",
                  cmap = "magma", lims = c(0,0.8))

# Right: with opponent adjustment (wide spread)
p4 <- plot_slopes(sum_slopes4, fit3, "With opponent adjustment",
                  cmap = "viridis", lims = c(0,1))
p3 + p4
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
  seed = 123,
  file = "fits/fit4"
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
  seed = 123,
  file = "fits/fit5"
)
sum_slopes5 <- make_sum_slopes(fit4)   # without opponent adjustment
sum_slopes6 <- make_sum_slopes(fit5)   # with opponent adjustment
p5 <- plot_slopes(sum_slopes5, fit4, "Without opponent adjustment",
                  cmap = "magma", lims = c(0,0.8))
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



fits <- list(fit1 = fit, fit2 = fit2, fit3 = fit4)
fits_adj <- list(fit1 = fit1, fit2 = fit3)  # only two adjusted

for (i in 1:3) {
  file_name <- paste0("brms_models", i, ".docx")
  
  if (i <= 2) {
    # paired comparison
    msummary(
      list("Baseline" = fits[[i]], "Opponent-adjusted" = fits_adj[[i]]),
      statistic = "[{conf.low}, {conf.high}]",
      output = file_name
    )
  } else {
    # baseline only
    msummary(
      list("Baseline" = fits[[i]]),
      statistic = "[{conf.low}, {conf.high}]",
      output = file_name
    )
  }
}

library(patchwork)

(p1 + p2) +
  plot_annotation(
    subtitle = "Each line shows an individual player’s pressure effect (logit scale). Black Line = median estimate, bars = 95% interval, dashed line = population-level effect."
  )

(p3 + p4) +
  plot_annotation(
    subtitle = "Each line shows an individual player’s pressure effect (logit scale). Black Line = median estimate, bars = 95% interval, dashed line = population-level effect."
  )

p5 +
  plot_annotation(
    subtitle = "Each line shows an individual player’s pressure effect (logit scale). Black Line = median estimate, bars = 95% interval, dashed line = population-level effect."
  )


# this will help us format the labels on the secondary y-axis
my_format <- function(number) {
  formatC(number, digits = 2, format = "f")
}

# grab the theta_j summaries
groups <-
  coef(fit3)$player_id[, , "pressure"] %>% 
  data.frame() %>% 
  mutate(player_id = as.numeric(rownames(.))) %>%
  arrange(Estimate)

# grab the mu summary
average <-
  fixef(fit3)["pressure",]

# combine and wrangle
post <-
  bind_rows(groups, average) %>% 
  mutate(rank     = c(1:112, 0),
         Estimate = exp(Estimate),
         Q2.5     = exp(Q2.5),
         Q97.5    = exp(Q97.5)) %>% 
  left_join(inter, by = "player_id") %>% 
  arrange(rank) %>% 
  mutate(label   = ifelse(is.na(Player), "POPULATION AVERAGE", Player),
         summary = str_c(my_format(Estimate), " [", my_format(Q2.5), ", ", my_format(Q97.5), "]"))

# what have we done?
post %>% 
  glimpse()

post <-
  post %>%
  distinct(player_id, .keep_all = TRUE)

post %>% 
  ggplot(aes(x = Estimate, xmin = Q2.5, xmax = Q97.5, y = rank)) +
  geom_interval(aes(color = label == "POPULATION AVERAGE"),
                size = 1/2) +
  geom_point(aes(size = 1 - Est.Error, color = label == "POPULATION AVERAGE"),
             shape = 15) +
  scale_color_viridis_d(option = "C", begin = .33, end = .67) +
  scale_size_continuous(range = c(1, 3.5)) +
  scale_x_continuous("odds ratio", breaks = 1:6 * 10, expand = expansion(mult = c(0.005, 0.005))) +
  scale_y_continuous(NULL, breaks = 0:112, limits = c(-1, 113), expand = c(0, 0),
                     labels = pull(post, label),
                     sec.axis = dup_axis(labels = pull(post, summary))) +
  theme(text = element_text(family = "Times"),
        axis.text.y = element_text(hjust = 0, color = "white", size = 6),
        axis.text.y.right = element_text(hjust = 1, size = 6),
        axis.ticks.y = element_blank(),
        panel.background = element_rect(fill = "grey8"),
        panel.border = element_rect(color = "transparent"))
