
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss, roc_auc_score, roc_curve, RocCurveDisplay
)
import pandas as pd

def evaluate_and_plot(model_dict, X_test, y_test, role_name):
    results = []
    plt.figure(figsize=(8, 6))

    for name, model, mtype in model_dict:
        if mtype == 'keras':
            y_proba = model.predict(X_test).flatten()
            y_pred = (y_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logloss = log_loss(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'Log Loss': logloss,
            'Brier Score': brier,
            'AUC-ROC': auc
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison ({role_name} Models)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'evaluation/roc_curve_{role_name.lower()}.png')
    plt.show()

    return pd.DataFrame(results).set_index("Model")
