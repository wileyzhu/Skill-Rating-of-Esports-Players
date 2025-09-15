# Skill Rating of Esports Players in League of Legends

A comprehensive machine learning research project analyzing professional League of Legends player performance and skill rating prediction across different roles. This dissertation project combines web scraping, feature engineering, and multiple ML models to predict player skill ratings based on in-game performance metrics.

## Project Overview

This research investigates how different machine learning models can predict esports player skill ratings using professional match data from League of Legends tournaments. The project analyzes performance across five distinct roles (Top, Jungle, Mid, ADC, Support) and compares multiple modeling approaches including traditional ML algorithms and neural networks.

## Key Features

### Data Collection & Processing
- **Web Scraping**: Automated data collection from [Gol.gg](https://gol.gg) for professional LoL matches
- **Match Data**: Comprehensive player statistics from Tier 1 tournaments (Worlds, LEC, LCK, etc.)
- **Feature Engineering**: Advanced statistical features and role-specific metrics
- **Series Detection**: Automatic Bo3/Bo5 series identification and clutch performance analysis

### Machine Learning Models
- **Logistic Regression**: Baseline linear model with regularization
- **Random Forest**: Ensemble method for feature importance analysis
- **XGBoost**: Gradient boosting for high-performance predictions
- **Neural Networks**: Deep learning approach with TensorFlow/Keras
- **Bayesian Modeling**: Statistical inference using R

### Analysis & Evaluation
- **Role-Specific Models**: Separate models optimized for each position
- **Comprehensive Metrics**: Accuracy, F1-score, AUC-ROC, Brier Score, Log Loss
- **Model Calibration**: Reliability analysis of prediction confidence
- **Feature Importance**: SHAP analysis for model interpretability
- **Clutch Performance**: Statistical analysis of high-pressure situations

## Project Structure

```
├── src/                        # Main source code modules
│   ├── main.py                # Complete pipeline orchestration
│   ├── data_collection.py     # Web scraping and data collection
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── model_training.py      # ML model training with hyperparameter tuning
│   ├── model_evaluation.py    # Comprehensive model evaluation
│   └── shap_analysis.py       # Model interpretability analysis
├── models/                     # Trained model artifacts
│   ├── best_*_model_*.pkl     # Scikit-learn models
│   ├── best_*_model.h5        # Neural network models
│   └── *_scaler.pkl           # Feature scalers
├── results/                    # Evaluation results and metrics
│   ├── metrics_*.csv          # Performance metrics by role
│   ├── performance_summary.csv # Best model summary
│   └── classification_reports/ # Detailed classification reports
├── shap_plots/                # Model interpretability visualizations
├── figures/                   # Research visualizations
├── clutch.R                   # Bayesian clutch performance analysis
└── *.pdf                      # Research papers and references
```

## Requirements

### Python Environment
```bash
# Core dependencies
pip install pandas numpy scikit-learn tensorflow xgboost
pip install selenium webdriver-manager
pip install matplotlib seaborn plotly shap
pip install scikeras joblib

# Optional for enhanced analysis
pip install optuna hyperopt bayesian-optimization
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd lol-skill-rating-analysis

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python src/main.py --skip-collection  # Uses existing data
```

### R Environment (for Bayesian analysis)
```r
install.packages(c("rstanarm", "bayesplot", "ggplot2", "dplyr"))
```

### System Requirements
- Python 3.8+
- R 4.0+ (for Bayesian modeling)
- Google Chrome + ChromeDriver (for web scraping)
- 8GB+ RAM recommended for model training

## Usage

### Quick Start - Complete Pipeline
```bash
# Run the entire analysis pipeline
python src/main.py

# Or with custom options
python src/main.py --skip-collection --log-level DEBUG
```

### Individual Components

#### 1. Data Collection
```python
from src.data_collection import MatchScraper
import numpy as np

# Initialize scraper
scraper = MatchScraper(headless=True)

# Scrape specific game ID range
game_ids = np.arange(53697, 54000)
df = scraper.scrape_games(game_ids, "output.csv")
```

#### 2. Data Preprocessing
```python
from src.data_preprocessing import DataProcessor

# Initialize processor
processor = DataProcessor()

# Run complete preprocessing pipeline
df, role_datasets = processor.process_full_pipeline()
```

#### 3. Model Training
```python
from src.model_training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Train models for a specific role
X = role_data[feature_names]
y = role_data['Win']
results = trainer.train_role_models(X, y, "Top")
```

#### 4. Model Evaluation
```python
from src.model_evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Run comprehensive evaluation
evaluation_results = evaluator.run_comprehensive_evaluation(trainer_results)
```

#### 5. SHAP Analysis
```python
from src.shap_analysis import SHAPAnalyzer

# Initialize analyzer
analyzer = SHAPAnalyzer()

# Run interpretability analysis
analyzer.run_complete_shap_analysis(trainer_results, feature_names)
```

#### 6. Bayesian Analysis (R)
```r
# Run clutch performance analysis
source("clutch.R")
```

## Key Findings

- **Role-Specific Performance**: Different roles require distinct modeling approaches with varying optimal features
- **Feature Importance**: KDE (Kill-Death-Assist efficiency), GPM, and early game differentials are key predictors
- **Model Comparison**: XGBoost consistently outperforms other approaches across most roles
- **Clutch Factor**: Bayesian analysis reveals significant performance differences in high-pressure Bo5 situations
- **Model Interpretability**: SHAP analysis shows role-specific feature importance patterns
- **Calibration**: Neural networks demonstrate better probability calibration than tree-based models

## Model Performance

| Role    | Best Model | Accuracy | F1-Score | AUC-ROC |
|---------|------------|----------|----------|---------|
| Top     | XGBoost    | 0.896    | 0.896    | 0.963   |
| Jungle  | XGBoost    | 0.887    | 0.888    | 0.910   |
| Mid     | XGBoost    | 0.893    | 0.894    | 0.907   |
| ADC     | XGBoost    | 0.890    | 0.891    | 0.912   |
| Support | XGBoost    | 0.888    | 0.889    | 0.909   |

## Research Applications

- **Talent Scouting**: Automated identification of promising players using ML-based skill ratings
- **Team Strategy**: Data-driven role-specific performance optimization and draft analysis
- **Tournament Analysis**: Real-time match outcome prediction and player performance forecasting
- **Esports Analytics**: Advanced statistical modeling framework adaptable to other competitive games
- **Academic Research**: Methodological contributions to sports analytics and machine learning

## Academic Context

This project serves as a dissertation research investigating the application of machine learning techniques to esports performance analysis. It contributes to the growing field of sports analytics by adapting traditional statistical methods to the unique characteristics of competitive gaming.

## Data Sources

- **Primary**: Gol.gg professional match database
- **Scope**: Tier 1 tournaments (2020-2024)
- **Sample Size**: 10,000+ professional matches
- **Players**: 500+ unique professional players

## Limitations

- Data limited to professional matches (may not generalize to amateur play)
- Role meta changes over time may affect model stability
- Sample size varies significantly between roles
- Potential selection bias in tournament participation

## Future Work

- Real-time prediction system integration
- Cross-game esports performance modeling
- Temporal analysis of skill development
- Integration with additional data sources (solo queue, scrimmages)

## Module Documentation

### Core Modules

#### `src/main.py`
Complete pipeline orchestration with command-line interface. Supports running individual phases or the complete analysis workflow.

**Key Features:**
- Configurable pipeline execution
- Comprehensive logging
- Error handling and recovery
- Final report generation

#### `src/data_collection.py`
Web scraping module for collecting professional match data from Gol.gg.

**Key Features:**
- Selenium-based automated scraping
- Batch processing for large datasets
- Error handling for missing/invalid games
- Team and player statistics extraction

#### `src/data_preprocessing.py`
Data cleaning, feature engineering, and dataset preparation.

**Key Features:**
- Bo5 series identification
- Advanced feature engineering (KDE, Multi-Kill indicators)
- Role-specific dataset preparation
- Data quality validation

#### `src/model_training.py`
Machine learning model training with hyperparameter optimization.

**Key Features:**
- Multiple model types (Logistic, RF, XGBoost, Neural Networks)
- Randomized hyperparameter search
- Cross-validation
- Model persistence

#### `src/model_evaluation.py`
Comprehensive model evaluation with multiple metrics and visualizations.

**Key Features:**
- Multiple evaluation metrics (Accuracy, F1, AUC-ROC, Brier Score)
- ROC and calibration curves
- Confusion matrices and classification reports
- Performance comparison visualizations

#### `src/shap_analysis.py`
Model interpretability analysis using SHAP values.

**Key Features:**
- SHAP value calculation for all model types
- Feature importance analysis
- Dependence plots
- Multi-role comparison visualizations

## Command Line Usage

```bash
# Run complete pipeline with existing data
python src/main.py --skip-collection

# Run only training and evaluation
python src/main.py --skip-collection --skip-shap

# Debug mode with verbose logging
python src/main.py --log-level DEBUG

# Custom data path
python src/main.py --base-path /path/to/data --skip-collection
```

## Output Files

The pipeline generates several types of output files:

- **Models**: `models/best_{role}_model_{type}.pkl` - Trained model artifacts
- **Metrics**: `results/metrics_{role}.csv` - Detailed performance metrics
- **Visualizations**: `results/*.png` - Performance comparison plots
- **SHAP Plots**: `shap_plots/*.png` - Model interpretability visualizations
- **Reports**: `analysis_report.txt` - Final analysis summary

## License

This research project is for academic purposes. Match data is publicly available through Gol.gg. Please cite appropriately if using this work for research purposes.

