import os
import pathlib
os.chdir(pathlib.Path(__file__).parents[1]/ 'dissertation')
import joblib

# Load the best neural network models
from tensorflow.keras.models import load_model
top_model = load_model('best_top_model.h5')
mid_model = load_model('best_mid_model.h5')
jungle_model = load_model('best_jungle_model.h5')
adc_model = load_model('best_adc_model.h5')
support_model = load_model('best_support_model.h5')

# Load the best Logistic Regression models
top_logistic = joblib.load('best_top_model_logistic.pkl')
mid_logistic = joblib.load('best_mid_model_logistic.pkl')
jungle_logistic = joblib.load('best_jungle_model_logistic.pkl')
adc_logistic = joblib.load('best_adc_model_logistic.pkl')
support_logistic = joblib.load('best_support_model_logistic.pkl')

# Load the best Random Forest models
top_rf = joblib.load('best_top_model_random_forest.pkl')
mid_rf = joblib.load('best_mid_model_random_forest.pkl')
jungle_rf = joblib.load('best_jungle_model_random_forest.pkl')
adc_rf = joblib.load('best_adc_model_random_forest.pkl')
support_rf = joblib.load('best_support_model_random_forest.pkl')

# Load the best XGBoost models
top_xgb = joblib.load('best_top_model_xgboost.pkl')
mid_xgb = joblib.load('best_mid_model_xgboost.pkl')
jungle_xgb = joblib.load('best_jungle_model_xgboost.pkl')
adc_xgb = joblib.load('best_adc_model_xgboost.pkl')
support_xgb = joblib.load('best_support_model_xgboost.pkl')

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb

# ---------------- KERAS HELPERS ----------------
def _lr_to_float(lr):
    """Handle keras learning rate objects → float."""
    try:
        import tensorflow as tf
        if isinstance(lr, (int, float, np.number)):
            return float(lr)
        if isinstance(lr, (tf.Variable, tf.Tensor)):
            return float(tf.keras.backend.get_value(lr))
        if hasattr(lr, "get_config"):
            cfg = lr.get_config()
            for k in ("initial_learning_rate", "learning_rate"):
                if k in cfg:
                    return float(cfg[k])
    except Exception:
        pass
    return None

def summarize_keras_model(name, model):
    row = {
        "Role": name,
        "Layers": None,
        "Total_params": None,
        "Trainable_params": None,
        "Non_trainable_params": None,
        "Optimizer": None,
        "Loss": None
    }

    # Architecture string
    try:
        parts = []
        for layer in getattr(model, "layers", []):
            cfg = layer.get_config()
            lname = layer.__class__.__name__
            if lname == "Dense":
                parts.append(f"Dense({cfg.get('units')}, {cfg.get('activation')})")
            elif lname == "Dropout":
                parts.append(f"Dropout({cfg.get('rate')})")
            else:
                parts.append(lname)
        row["Layers"] = " → ".join(parts) if parts else None
    except Exception:
        pass

    # Params
    try:
        row["Total_params"] = int(model.count_params())
        row["Trainable_params"] = sum(int(w.shape.num_elements()) for w in model.trainable_weights)
        row["Non_trainable_params"] = sum(int(w.shape.num_elements()) for w in model.non_trainable_weights)
    except Exception:
        pass

    # Optimizer / Loss
    opt = getattr(model, "optimizer", None)
    if opt is not None:
        row["Optimizer"] = type(opt).__name__
    loss = getattr(model, "loss", None)
    if loss is not None:
        row["Loss"] = getattr(loss, "__name__", str(loss))

    return row

# ---------------- SKLEARN / XGB HELPERS ----------------
def unwrap_estimator(model):
    """Unwrap pipelines, CV, calibration to core estimator."""
    ctx = {}
    est = model
    if isinstance(est, (GridSearchCV, RandomizedSearchCV)):
        ctx["best_params_"] = getattr(est, "best_params_", None)
        est = est.best_estimator_
    if isinstance(est, Pipeline):
        ctx["pipeline_steps"] = [n for n, _ in est.steps]
        est = est.steps[-1][1]
    if isinstance(est, CalibratedClassifierCV):
        ctx["calibration_method"] = est.method
        est = est.base_estimator
    return est, ctx

def summarize_sklearn(name, model):
    est, ctx = unwrap_estimator(model)
    merged = getattr(est, "get_params", lambda **k: {})()
    if ctx.get("best_params_"):
        best = {k.split("__")[-1]: v for k,v in ctx["best_params_"].items()}
        merged.update(best)

    row = {"Role": name, "Estimator": type(est).__name__}

    if "logistic" in type(est).__name__.lower():
        for k in ["penalty","solver","C","max_iter","fit_intercept"]:
            row[k] = merged.get(k)

    elif "Forest" in type(est).__name__:
        for k in ["n_estimators","max_depth","min_samples_split","min_samples_leaf",
                  "max_features","bootstrap","random_state"]:
            row[k] = merged.get(k)

    elif isinstance(est, (xgb.XGBClassifier, xgb.XGBRegressor)):
        try:
            xgbp = est.get_xgb_params()
        except Exception:
            xgbp = {}
        for k in ["n_estimators","max_depth","learning_rate","subsample","colsample_bytree",
                  "gamma","reg_lambda","reg_alpha"]:
            row[k] = xgbp.get(k, merged.get(k))

    else:
        for k,v in merged.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                row[k] = v
    return row

# ---------------- BUILD TABLES ----------------
roles = ["Top","Mid","Jungle","ADC","Support"]

# Neural Nets
keras_models = {
    "Top": top_model, "Mid": mid_model, "Jungle": jungle_model, "ADC": adc_model, "Support": support_model
}
df_keras = pd.DataFrame([summarize_keras_model(n, m) for n,m in keras_models.items()]).set_index("Role").reindex(roles)

# Logistic
log_models = {
    "Top": top_logistic, "Mid": mid_logistic, "Jungle": jungle_logistic, "ADC": adc_logistic, "Support": support_logistic
}
df_log = pd.DataFrame([summarize_sklearn(n,m) for n,m in log_models.items()]).set_index("Role").reindex(roles)

# Random Forest
rf_models = {
    "Top": top_rf, "Mid": mid_rf, "Jungle": jungle_rf, "ADC": adc_rf, "Support": support_rf
}
df_rf = pd.DataFrame([summarize_sklearn(n,m) for n,m in rf_models.items()]).set_index("Role").reindex(roles)

# XGBoost
xgb_models = {
    "Top": top_xgb, "Mid": mid_xgb, "Jungle": jungle_xgb, "ADC": adc_xgb, "Support": support_xgb
}
df_xgb = pd.DataFrame([summarize_sklearn(n,m) for n,m in xgb_models.items()]).set_index("Role").reindex(roles)

# ---------------- SAVE OUTPUT ----------------
with pd.ExcelWriter("appendix_model_specs.xlsx") as xl:
    df_keras.to_excel(xl, sheet_name="NeuralNet")
    df_log.to_excel(xl,   sheet_name="Logistic")
    df_rf.to_excel(xl,    sheet_name="RandomForest")
    df_xgb.to_excel(xl,   sheet_name="XGBoost")

print("\nNeural Nets:\n", df_keras.head())
print("\nLogistic:\n", df_log.head())
print("\nRandom Forest:\n", df_rf.head())
print("\nXGBoost:\n", df_xgb.head())