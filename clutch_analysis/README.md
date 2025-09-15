# Clutch Performance Analysis in League of Legends Esports

This folder contains a comprehensive Bayesian analysis of clutch performance in professional League of Legends, examining how players perform under high-pressure situations.

## 📊 Analysis Overview

This study investigates **clutch performance** - the ability to maintain or improve performance under pressure - using Bayesian hierarchical modeling. We examine three types of pressure situations:

1. **Playoff vs Regular Season** games
2. **International vs Domestic** tournaments  
3. **Final games in Bo5 series** (clutch moments)

## 🔬 Methodology

### Bayesian Hierarchical Models
- **Family**: Zero-one inflated beta regression
- **Structure**: Player-specific random effects for pressure response
- **Priors**: Weakly informative priors for robust inference
- **Software**: R with `brms` package

### Model Specifications
```r
performance_score ~ pressure + (pressure | player_id) + (1 | opponent_team)
```

Where:
- `performance_score`: Normalized player performance (0-1 scale)
- `pressure`: Binary indicator for high-pressure situation
- `player_id`: Random effects for individual players
- `opponent_team`: Random effects for opponent strength

## 📁 File Structure

```
clutch_analysis/
├── clutch.R                    # Main analysis script
├── README.md                   # This documentation
├── ANALYSIS_SUMMARY.md         # Key findings and results
├── literature/                 # Academic references
│   ├── 1-s2.0-S0167268119303087-main.pdf
│   ├── 1-s2.0-S1469029221000078-main.pdf
│   ├── Clutch performance in sport and exercise  a systematic review.pdf
│   ├── Clutch Tennis.pdf
│   ├── QualExplorationofchoking.JCSP.pdf
│   ├── SSRN-id1624975.pdf
│   └── zheng-cao-et-al-2011-performance-under-pressure-in-the-nba.pdf
└── Literture Review.docx       # Literature review document
```

## 🎯 Key Research Questions

1. **Do players perform differently under pressure?**
   - Population-level pressure effects
   - Individual variation in pressure response

2. **Which players are truly "clutch"?**
   - Player-specific pressure coefficients
   - Probability of positive pressure effect

3. **How does context matter?**
   - Playoff vs regular season
   - International vs domestic competition
   - Final games in series

## 📈 Analysis Pipeline

### 1. Data Preparation
```r
# Filter players with both pressure/non-pressure games
players_both <- d %>%
  group_by(player_id) %>%
  summarise(n_playoff = sum(State == "Playoff"),
            n_regular = sum(State == "Regular")) %>%
  filter(n_playoff > 0 & n_regular > 0)
```

### 2. Model Fitting
```r
# Fit Bayesian hierarchical model
fit <- brm(
  bf(performance_score ~ pressure + (pressure | player_id)),
  family = zero_one_inflated_beta(),
  data = playoff,
  chains = 4, cores = 4,
  iter = 2000, warmup = 1000
)
```

### 3. Player-Specific Effects
```r
# Extract individual player pressure effects
sum_slopes <- tot_slopes %>%
  group_by(player) %>%
  summarise(
    med = median(beta_player),
    lo  = quantile(beta_player, .025),
    hi  = quantile(beta_player, .975),
    pr_pos = mean(beta_player > 0)
  )
```

## 🔍 Model Comparisons

The analysis includes multiple model specifications:

| Model | Pressure Context | Opponent Adjustment | Players | Key Finding |
|-------|------------------|-------------------|---------|-------------|
| fit   | Playoff vs Regular | No | ~200 | Small population effect |
| fit1  | Playoff vs Regular | Yes | ~200 | Stronger individual variation |
| fit2  | International vs Domestic | No | ~112 | Moderate pressure effects |
| fit3  | International vs Domestic | Yes | ~112 | Clear clutch players emerge |
| fit4  | Bo5 Final Games | No | ~80 | High-stakes performance |
| fit5  | Bo5 Final Games | Yes | ~80 | Most extreme clutch effects |

## 📊 Visualization Outputs

The analysis generates several key visualizations:

1. **Player-Specific Pressure Effects**: Forest plots showing individual clutch coefficients
2. **Population vs Individual Effects**: Comparison of average vs player-specific responses
3. **Odds Ratio Plots**: Interpretable effect sizes for clutch performance
4. **Model Comparison Plots**: Side-by-side comparisons of different pressure contexts

## 🎯 Key Findings

### Population-Level Effects
- **Playoff Pressure**: Small negative effect (most players perform slightly worse)
- **International Pressure**: Moderate positive effect (players rise to occasion)
- **Bo5 Final Games**: Strong positive effect (true clutch moments)

### Individual Variation
- **High Heterogeneity**: Players vary dramatically in pressure response
- **Clutch Players**: ~10-20% of players show consistent positive pressure effects
- **Context Matters**: Same players may be clutch in some contexts but not others

### Statistical Robustness
- **Opponent Adjustment**: Including opponent strength reveals stronger individual effects
- **Model Fit**: Zero-one inflated beta handles performance score distribution well
- **Convergence**: All models show good MCMC diagnostics

## 🔧 Technical Requirements

### R Packages
```r
library(brms)        # Bayesian modeling
library(ggplot2)     # Visualization
library(tidyverse)   # Data manipulation
library(ggdist)      # Distribution plots
library(patchwork)   # Plot composition
library(modelsummary) # Model tables
```

### Computational Requirements
- **Memory**: 8GB+ RAM recommended
- **Time**: 2-4 hours for full analysis
- **Cores**: 4+ cores for parallel MCMC chains

## 📚 Academic Context

This analysis contributes to the sports psychology literature on clutch performance by:

1. **Methodological Innovation**: First Bayesian analysis of esports clutch performance
2. **Individual Differences**: Quantifying player-specific pressure responses
3. **Context Specificity**: Showing clutch performance varies by situation type
4. **Esports Application**: Extending clutch research to competitive gaming

## 🚀 Usage

### Running the Complete Analysis
```r
source("clutch_analysis/clutch.R")
```

### Key Functions
```r
# Create player-specific slope summaries
sum_slopes <- make_sum_slopes(fit)

# Generate comparison plots
plot_slopes(sum_slopes, fit, "Title", "viridis", c(0,1))
```

## 📖 References

The `literature/` folder contains key academic papers on clutch performance in sports, providing theoretical foundation for the esports analysis.

## 🎓 Integration with Main Project

This clutch analysis complements the main machine learning pipeline by:
- Providing deeper insight into high-pressure performance
- Identifying players with exceptional clutch abilities
- Informing feature engineering for pressure-sensitive metrics
- Supporting talent evaluation and team strategy decisions