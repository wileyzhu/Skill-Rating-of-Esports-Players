# Model Inventory

This directory contains all trained models for the League of Legends Player Skill Rating project.

## Model Organization

### By Role
- **Top Lane**: 4 models (Logistic, Random Forest, XGBoost, Neural Network)
- **Jungle**: 4 models (Logistic, Random Forest, XGBoost, Neural Network)  
- **Mid Lane**: 4 models (Logistic, Random Forest, XGBoost, Neural Network)
- **ADC**: 4 models (Logistic, Random Forest, XGBoost, Neural Network)
- **Support**: 4 models (Logistic, Random Forest, XGBoost, Neural Network)

### Model Types

#### Scikit-learn Models (.pkl files)
- `best_{role}_model_logistic.pkl` - Logistic Regression with L1/L2 regularization
- `best_{role}_model_random_forest.pkl` - Random Forest ensemble classifier
- `best_{role}_model_xgboost.pkl` - XGBoost gradient boosting classifier

#### Neural Network Models (.h5 files)
- `best_{role}_model.h5` - TensorFlow/Keras neural network models

## Model Loading Examples

### Loading Scikit-learn Models
```python
import joblib

# Load a specific model
model = joblib.load('models/best_top_model_xgboost.pkl')

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Loading Neural Network Models
```python
from tensorflow import keras

# Load neural network
model = keras.models.load_model('models/best_top_model.h5')

# Make predictions
predictions = model.predict(X_test)
```

## Model Performance Summary

Based on the results files, here are the best performing models per role:

| Role    | Best Model Type | Key Metrics Available |
|---------|----------------|----------------------|
| Top     | XGBoost        | See results/metrics_Top.csv |
| Jungle  | XGBoost        | See results/metrics_Jungle.csv |
| Mid     | Neural Network | See results/metrics_Mid.csv |
| ADC     | XGBoost        | See results/metrics_ADC.csv |
| Support | Random Forest  | See results/metrics_Support.csv |

## Feature Requirements

All models expect the following features in order:
1. KDE (Kill-Death-Assist Efficiency)
2. DPM (Damage Per Minute)
3. Multi-Kill (Binary indicator)
4. GPM (Gold Per Minute)
5. VSPM (Vision Score Per Minute)
6. WCPM (Wards Cleared Per Minute)
7. GD@15 (Gold Difference at 15 minutes)
8. XPD@15 (Experience Difference at 15 minutes)
9. CSD@15 (Creep Score Difference at 15 minutes)
10. LVLD@15 (Level Difference at 15 minutes)
11. DTPD (Damage Taken Per Death)

## Usage with Pipeline

These models are automatically loaded and used by the main pipeline:

```python
from src.main import LoLAnalysisPipeline

pipeline = LoLAnalysisPipeline()
# Models will be loaded from this directory automatically
```

## Model Versioning

- All models were trained with the same hyperparameter tuning approach
- Random state: 42 (for reproducibility)
- Cross-validation: 3-fold
- Hyperparameter search: RandomizedSearchCV with 10-20 iterations per model type

## File Sizes

Total model directory size: ~50MB
- Neural networks (.h5): ~2-5MB each
- Scikit-learn models (.pkl): ~1-10MB each depending on complexity