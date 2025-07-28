import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_shap_summary(models, role_names, x_data_dict, feature_names, num_features=10):
    shap.initjs()
    shap_values_dict = {}
    feature_values_dict = {}

    for model, x, role in zip(models, x_data_dict.values(), role_names):
        sampled_df = x.sample(100, random_state=42)
        explainer = shap.KernelExplainer(model.predict, sampled_df.values.astype('float32'))
        shap_values = explainer.shap_values(sampled_df.values.astype('float32'))
        shap_values_dict[role] = np.squeeze(shap_values)
        feature_values_dict[role] = sampled_df

    mean_abs_shap = np.mean([np.abs(shap_values_dict[role]) for role in role_names], axis=0)
    feature_order = np.argsort(mean_abs_shap.mean(axis=0))[::-1][:num_features]

    for role in role_names:
        shap_values = shap_values_dict[role][:, feature_order]
        feature_values = feature_values_dict[role].iloc[:, feature_order]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, features=feature_values.values,
                          feature_names=np.array(feature_names)[feature_order],
                          max_display=num_features, plot_type='dot', show=False)
        plt.title(f'SHAP Summary Plot for {role}')
        plt.savefig(f'shap_summary_{role}.png')
        plt.close()