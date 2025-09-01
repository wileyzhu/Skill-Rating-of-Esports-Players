# -*- coding: utf-8 -*-
"""
Recreate 3 figures from per-role CSVs that have no model column.
Row order in each CSV is assumed to be:
0: Logistic Regression, 1: Random Forest, 2: XGBoost, 3: Neural Network
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config ----
os.chdir('/Users/wiley/Desktop/Dissertation')
roles   = ["Top", "Jungle", "Mid", "ADC", "Support"]
models4 = ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"]

# helper to normalise column names like "AUC-ROC", "AUC_ROC", "auc roc" → "auc-roc"
def normalise_cols(df):
    m = {c: c.lower().replace('_',' ').replace('-', ' ').replace('%','pct') for c in df.columns}
    df2 = df.copy()
    df2.columns = list(m.values())

    # map a few common variants to canonical keys we’ll use below
    rename = {
        "auc roc": "auc-roc",
        "roc auc": "auc-roc",
        "auc": "auc-roc",
        "log loss": "log loss",
        "brier score": "brier score",
        "accuracy": "accuracy",
        "f1 score": "f1-score",
        "f1": "f1-score"
    }
    df2.columns = [rename.get(col, col) for col in df2.columns]
    return df2

# read each CSV, enforce row order, and grab metric columns
metric_mats = {  # will end up as np.array with shape (len(roles), len(models4))
    "auc-roc": [],
    "log loss": [],
    "brier score": [],
    "accuracy": [],
    "f1-score": []
}

for role in roles:
    df = pd.read_csv(f"metrics_{role}.csv")
    df = normalise_cols(df)

    # Validate row count (should be 4: LR, RF, XGB, NN)
    if len(df) < 4:
        raise ValueError(f"{role}: expected at least 4 rows (LR, RF, XGB, NN). Found {len(df)}.")

    # Extract in fixed order by row index
    # If your file sometimes has more rows, we just take the first 4
    df = df.iloc[:4, :]

    # Fetch values for each metric; will raise KeyError if missing
    try:
        metric_mats["auc-roc"].append(df["auc-roc"].to_numpy(dtype=float))
        metric_mats["log loss"].append(df["log loss"].to_numpy(dtype=float))
        metric_mats["brier score"].append(df["brier score"].to_numpy(dtype=float))
        metric_mats["accuracy"].append(df["accuracy"].to_numpy(dtype=float))
        metric_mats["f1-score"].append(df["f1-score"].to_numpy(dtype=float))
    except KeyError as e:
        raise KeyError(f"{role}: missing required metric column: {e}. "
                       f"Columns seen: {list(df.columns)}")

# Stack to (roles x models)
for k in metric_mats:
    metric_mats[k] = np.vstack(metric_mats[k])  # shape (5 roles, 4 models)

# Small helper
def save_tight(fig, path, dpi=220):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

# ---------------------------------------------------------------
# 1) AUC-ROC heatmap
# ---------------------------------------------------------------
auc = metric_mats["auc-roc"]

fig, ax = plt.subplots(figsize=(10, 6))
# set a tight color range around your values; tweak if needed
vmin = max(0.90, auc.min() - 0.002)
vmax = min(0.99, auc.max() + 0.002)
im = ax.imshow(auc, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")

ax.set_xticks(np.arange(len(models4)))
ax.set_xticklabels(models4, rotation=20, ha="right")
ax.set_yticks(np.arange(len(roles)))
ax.set_yticklabels(roles)
ax.set_title("Model Comparison by Role (AUC-ROC)", fontsize=16, pad=10)

# annotate
for i in range(auc.shape[0]):
    for j in range(auc.shape[1]):
        ax.text(j, i, f"{auc[i, j]:.3f}", ha="center", va="center", color="black", fontsize=10)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("AUC-ROC")
save_tight(fig, "Model Comparison by Role (AUC-ROC).png")

# ---------------------------------------------------------------
# 2) Log Loss & Brier Score line charts
# ---------------------------------------------------------------
colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]  # LR, RF, XGB, NN

# Log loss
logloss = metric_mats["log loss"]
fig, ax = plt.subplots(figsize=(11, 5), nrows=1, ncols=2)
for j, m in enumerate(models4):
    ax[0].plot(roles, logloss[:, j], marker="o", linewidth=2.5, label=m, color=colors[j])
ax[0].set_ylabel("Log Loss")
ax[0].set_title("Log Loss by Role", fontsize=16, pad=8)
ax[0].legend(frameon=True)
ax[0].grid(axis="y", alpha=0.3)


# Brier
brier = metric_mats["brier score"]
for j, m in enumerate(models4):
    ax[1].plot(roles, brier[:, j], marker="o", linewidth=2.5, label=m, color=colors[j])
ax[1].set_ylabel("Brier Score")
ax[1].set_title("Brier Score by Role", fontsize=16, pad=8)
ax[1].legend(frameon=True)
ax[1].grid(axis="y", alpha=0.3)
save_tight(fig, "Calibration Metrics by Role.png")

# ---------------------------------------------------------------
# 3) Accuracy & F1 grouped bar charts
# ---------------------------------------------------------------
def grouped_bar(mat1, mat2, title1, title2, ylabel1, ylabel2, fname, ylim=None):
    n_roles, n_models = mat1.shape
    x = np.arange(n_roles)
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 5.5), nrows=1, ncols=2)
    for j in range(n_models):
        ax[0].bar(x + (j - (n_models-1)/2)*width, mat1[:, j], width=width, label=models4[j], color=colors[j])
        ax[1].bar(x + (j - (n_models-1)/2)*width, mat2[:, j], width=width, label=models4[j], color=colors[j])

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(roles)
    ax[0].set_ylabel(ylabel1)
    ax[0].set_title(title1, fontsize=16, pad=8)
    ax[0].legend(title="Model", ncol=1, frameon=True)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(roles)
    ax[1].set_ylabel(ylabel2)
    ax[1].set_title(title2, fontsize=16, pad=8)
    ax[1].legend(title="Model", ncol=1, frameon=True)

    if ylim is None:
        ymin1 = max(0.0, mat1.min() - 0.02)
        ymax1 = min(1.0, mat1.max() + 0.02)
        ymin2 = max(0.0, mat2.min() - 0.02)
        ymax2 = min(1.0, mat2.max() + 0.02)
        ax[0].set_ylim(ymin1, ymax1)
        ax[1].set_ylim(ymin2, ymax2)
    else:
        ax[0].set_ylim(*ylim)
        ax[1].set_ylim(*ylim)
    ax[0].grid(axis="y", alpha=0.3)
    save_tight(fig, fname)

accuracy = metric_mats["accuracy"]
f1 = metric_mats["f1-score"]
grouped_bar(accuracy, f1, "Accuracy by Role and Model", "F1-score by Role and Model", "Accuracy", "F1-score", "Accuracy and F1-score by Role and Model.png")

print("Saved figures:")
print(" - Model Comparison by Role (AUC-ROC).png")
print(" - Log Loss by Role.png")
print(" - Brier Score by Role.png")
print(" - Accuracy by Role and Model.png")
print(" - F1-score by Role and Model.png")