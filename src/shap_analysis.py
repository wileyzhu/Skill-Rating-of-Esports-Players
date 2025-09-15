"""
SHAP Analysis Module for League of Legends Player Skill Rating
Model interpretability and feature importance analysis using SHAP values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
from typing import Dict, List, Any, Optional
import joblib


class SHAPAnalyzer:
    """SHAP-based model interpretability analysis"""
    
    def __init__(self, models_dir: str = "models", output_dir: str = "shap_plots"):
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.shap_values_dict = {}
        self.feature_values_dict = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize SHAP
        shap.initjs()
    
    def calculate_shap_values(self, model: Any, X_sample: pd.DataFrame, 
                            model_type: str, role: str) -> np.ndarray:
        """Calculate SHAP values for a given model and dataset"""
        print(f"Calculating SHAP values for {role} {model_type}...")
        
        # Sample data for SHAP analysis (computational efficiency)
        if len(X_sample) > 100:
            X_sample = X_sample.sample(100, random_state=42)
        
        # Convert to numpy array for SHAP
        X_array = X_sample.values.astype('float32')
        
        try:
            if model_type == 'neural_network':
                # For neural networks, use DeepExplainer or KernelExplainer
                explainer = shap.KernelExplainer(model.predict, X_array[:50])  # Use subset as background
                shap_values = explainer.shap_values(X_array)
            elif model_type == 'xgboost':
                # Use TreeExplainer for XGBoost
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_array)
            elif model_type == 'random_forest':
                # Use TreeExplainer for Random Forest
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_array)
                # For binary classification, take positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:  # logistic regression
                # Use LinearExplainer for linear models
                explainer = shap.LinearExplainer(model, X_array)
                shap_values = explainer.shap_values(X_array)
            
            # Ensure 2D array
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]  # Take positive class for multi-output
            
            return np.squeeze(shap_values), X_sample
            
        except Exception as e:
            print(f"Error calculating SHAP values for {role} {model_type}: {e}")
            # Fallback to KernelExplainer
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') 
                else model.predict(x), 
                X_array[:20]
            )
            shap_values = explainer.shap_values(X_array)
            return np.squeeze(shap_values), X_sample
    
    def generate_summary_plots(self, role_models: Dict[str, Any], X_test: pd.DataFrame, 
                             role: str, feature_names: List[str]) -> None:
        """Generate SHAP summary plots for all models of a role"""
        
        for model_name, model in role_models.items():
            try:
                shap_values, X_sample = self.calculate_shap_values(model, X_test, model_name, role)
                
                # Store for later use
                self.shap_values_dict[f"{role}_{model_name}"] = shap_values
                self.feature_values_dict[f"{role}_{model_name}"] = X_sample
                
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values, X_sample, feature_names=feature_names,
                    show=False, max_display=len(feature_names)
                )
                plt.title(f'SHAP Summary - {role} {model_name.replace("_", " ").title()}')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/summary_{role.lower()}_{model_name}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Bar plot for feature importance
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values, X_sample, feature_names=feature_names,
                    plot_type="bar", show=False, max_display=len(feature_names)
                )
                plt.title(f'SHAP Feature Importance - {role} {model_name.replace("_", " ").title()}')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/importance_{role.lower()}_{model_name}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error generating SHAP plots for {role} {model_name}: {e}")
                continue
    
    def create_feature_importance_comparison(self, roles: List[str], 
                                           feature_names: List[str]) -> None:
        """Create comparison of feature importance across roles and models"""
        
        # Collect mean absolute SHAP values for each role-model combination
        importance_data = []
        
        for role in roles:
            for model_type in ['logistic', 'random_forest', 'xgboost', 'neural_network']:
                key = f"{role}_{model_type}"
                if key in self.shap_values_dict:
                    shap_values = self.shap_values_dict[key]
                    mean_importance = np.abs(shap_values).mean(axis=0)
                    
                    for i, feature in enumerate(feature_names):
                        importance_data.append({
                            'Role': role,
                            'Model': model_type.replace('_', ' ').title(),
                            'Feature': feature,
                            'Importance': mean_importance[i] if i < len(mean_importance) else 0
                        })
        
        if not importance_data:
            print("No SHAP data available for comparison plot")
            return
        
        # Create DataFrame and pivot for heatmap
        df_importance = pd.DataFrame(importance_data)
        
        # Create separate plots for each model type
        model_types = df_importance['Model'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, model_type in enumerate(model_types[:4]):  # Limit to 4 models
            if i < len(axes):
                model_data = df_importance[df_importance['Model'] == model_type]
                pivot_data = model_data.pivot(index='Feature', columns='Role', values='Importance')
                
                # Create heatmap
                import seaborn as sns
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                           ax=axes[i], cbar_kws={'label': 'Mean |SHAP Value|'})
                axes[i].set_title(f'Feature Importance - {model_type}')
                axes[i].set_xlabel('Role')
                axes[i].set_ylabel('Feature')
        
        # Remove empty subplots
        for j in range(len(model_types), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_role_comparison_plot(self, roles: List[str], feature_names: List[str],
                                  model_type: str = 'xgboost') -> None:
        """Create SHAP comparison plot across roles for a specific model"""
        
        # Collect SHAP values for the specified model across roles
        role_shap_data = {}
        role_feature_data = {}
        
        for role in roles:
            key = f"{role}_{model_type}"
            if key in self.shap_values_dict:
                role_shap_data[role] = self.shap_values_dict[key]
                role_feature_data[role] = self.feature_values_dict[key]
        
        if not role_shap_data:
            print(f"No SHAP data available for {model_type} across roles")
            return
        
        # Create multi-role comparison plot
        self._plot_multiple_shap_features_impact(
            role_shap_data, role_feature_data, list(role_shap_data.keys()),
            f"shap_comparison_{model_type}.png", feature_names
        )
    
    def _plot_multiple_shap_features_impact(self, shap_values_dict: Dict[str, np.ndarray],
                                          feature_values_dict: Dict[str, pd.DataFrame],
                                          roles: List[str], filename: str,
                                          feature_names: List[str], max_display: int = 10) -> None:
        """Create side-by-side SHAP beeswarm plots for multiple roles"""
        
        fig, axes = plt.subplots(nrows=1, ncols=len(roles), figsize=(4*len(roles), 8))
        if len(roles) == 1:
            axes = [axes]
        
        # Determine common feature order based on overall importance
        overall_importance = np.mean([
            np.abs(shap_values_dict[role]).mean(axis=0) for role in roles
        ], axis=0)
        feature_order = np.argsort(overall_importance)[::-1][:max_display]
        
        for idx, role in enumerate(roles):
            ax = axes[idx]
            shap_values = shap_values_dict[role][:, feature_order]
            feature_values = feature_values_dict[role].iloc[:, feature_order]
            
            # Create SHAP explanation object
            shap_explanation = shap.Explanation(
                values=shap_values,
                data=feature_values.values,
                feature_names=[feature_names[i] for i in feature_order]
            )
            
            # Create beeswarm plot
            shap.plots.beeswarm(shap_explanation, show=False, max_display=max_display)
            ax.set_title(f'{role}')
            
            # Only show y-labels for first plot
            if idx > 0:
                ax.set_yticklabels([])
            
            # Only show x-label for middle plot
            if idx != len(roles) // 2:
                ax.set_xlabel('')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_individual_feature_plots(self, role_models: Dict[str, Any], 
                                        X_test: pd.DataFrame, role: str,
                                        feature_names: List[str], top_features: int = 5) -> None:
        """Generate individual feature dependence plots"""
        
        for model_name, model in role_models.items():
            key = f"{role}_{model_name}"
            if key not in self.shap_values_dict:
                continue
            
            shap_values = self.shap_values_dict[key]
            X_sample = self.feature_values_dict[key]
            
            # Find top features by importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_feature_indices = np.argsort(feature_importance)[::-1][:top_features]
            
            # Create dependence plots for top features
            for i, feature_idx in enumerate(top_feature_indices):
                try:
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(
                        feature_idx, shap_values, X_sample,
                        feature_names=feature_names, show=False
                    )
                    plt.title(f'SHAP Dependence - {role} {model_name.replace("_", " ").title()} - {feature_names[feature_idx]}')
                    plt.tight_layout()
                    plt.savefig(
                        f'{self.output_dir}/dependence_{role.lower()}_{model_name}_{feature_names[feature_idx].replace("/", "_")}.png',
                        dpi=300, bbox_inches='tight'
                    )
                    plt.close()
                except Exception as e:
                    print(f"Error creating dependence plot for {feature_names[feature_idx]}: {e}")
                    continue
    
    def run_complete_shap_analysis(self, trainer_results: Dict[str, Dict],
                                 feature_names: List[str]) -> None:
        """Run complete SHAP analysis pipeline"""
        print("Starting SHAP analysis...")
        
        roles = list(trainer_results.keys())
        
        # Generate SHAP plots for each role
        for role, role_data in trainer_results.items():
            print(f"\nAnalyzing {role} models...")
            
            models = role_data['models']
            X_test, y_test = role_data['test_data']
            scaler = role_data['scaler']
            
            # Use scaled data for models that need it
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            X_test_scaled_df = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
            
            # Generate summary plots
            models_for_shap = {}
            for model_name, model in models.items():
                if model_name in ['logistic', 'neural_network']:
                    models_for_shap[model_name] = model
                    # Use scaled data
                    self.generate_summary_plots({model_name: model}, X_test_scaled_df, role, feature_names)
                else:
                    models_for_shap[model_name] = model
                    # Use original data
                    self.generate_summary_plots({model_name: model}, X_test_df, role, feature_names)
            
            # Generate individual feature plots
            self.generate_individual_feature_plots(models_for_shap, X_test_df, role, feature_names)
        
        # Create comparison plots
        self.create_feature_importance_comparison(roles, feature_names)
        
        # Create role comparison for best performing model
        self.create_role_comparison_plot(roles, feature_names, 'xgboost')
        
        print(f"\nSHAP analysis completed! Results saved to {self.output_dir}/")


def main():
    """Example usage of SHAPAnalyzer"""
    from model_training import ModelTrainer
    from data_preprocessing import DataProcessor
    
    # Load data and train models
    processor = DataProcessor()
    df, role_datasets = processor.process_full_pipeline()
    
    trainer = ModelTrainer()
    trainer_results = {}
    
    # Train models for each role
    for role, role_data in role_datasets.items():
        if len(role_data) >= 100:
            feature_names = processor.get_feature_names()
            X = role_data[feature_names]
            y = role_data['Win']
            trainer_results[role] = trainer.train_role_models(X, y, role)
    
    # Run SHAP analysis
    analyzer = SHAPAnalyzer()
    analyzer.run_complete_shap_analysis(trainer_results, processor.get_feature_names())


if __name__ == "__main__":
    main()