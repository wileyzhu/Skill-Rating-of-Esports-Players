"""
Model Evaluation Module for League of Legends Player Skill Rating
Comprehensive evaluation metrics and visualization for trained models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, calibration_curve
)
from sklearn.calibration import calibration_curve
import joblib
import os


class ModelEvaluator:
    """Comprehensive evaluation of trained models with metrics and visualizations"""
    
    def __init__(self, models_dir: str = "models", results_dir: str = "results"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.evaluation_results = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_proba),
            'log_loss': log_loss(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba)
        }
    
    def evaluate_model_predictions(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                 model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions based on model type"""
        if model_type == 'neural_network':
            y_proba = model.predict(X_test).flatten()
            y_pred = (y_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        return y_pred, y_proba
    
    def evaluate_role_models(self, role_models: Dict[str, Any], X_test: np.ndarray, 
                           y_test: np.ndarray, role: str) -> Dict[str, Dict[str, float]]:
        """Evaluate all models for a specific role"""
        role_results = {}
        
        for model_name, model in role_models.items():
            y_pred, y_proba = self.evaluate_model_predictions(model, X_test, y_test, model_name)
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_proba)
            role_results[model_name] = metrics
        
        self.evaluation_results[role] = role_results
        return role_results
    
    def plot_roc_curves(self, role_models: Dict[str, Any], X_test: np.ndarray, 
                       y_test: np.ndarray, role: str) -> None:
        """Plot ROC curves for all models of a role"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in role_models.items():
            y_pred, y_proba = self.evaluate_model_predictions(model, X_test, y_test, model_name)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {role} Role')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/roc_curves_{role.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_calibration_curves(self, role_models: Dict[str, Any], X_test: np.ndarray,
                              y_test: np.ndarray, role: str) -> None:
        """Plot calibration curves to assess prediction reliability"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in role_models.items():
            y_pred, y_proba = self.evaluate_model_predictions(model, X_test, y_test, model_name)
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_proba, n_bins=10
            )
            
            plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label=f'{model_name.replace("_", " ").title()}')
        
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted probability")
        plt.title(f'Calibration Curves - {role} Role')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/calibration_curves_{role.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_metrics_comparison_plot(self) -> None:
        """Create comprehensive metrics comparison across all roles and models"""
        # Prepare data for plotting
        plot_data = []
        for role, role_results in self.evaluation_results.items():
            for model, metrics in role_results.items():
                for metric, value in metrics.items():
                    plot_data.append({
                        'Role': role,
                        'Model': model.replace('_', ' ').title(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': value
                    })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create subplots for different metrics
        metrics_to_plot = ['Accuracy', 'F1 Score', 'Auc Roc', 'Log Loss', 'Brier Score']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes):
                metric_data = df_plot[df_plot['Metric'] == metric]
                
                # Create pivot table for heatmap
                pivot_data = metric_data.pivot(index='Role', columns='Model', values='Value')
                
                # For log loss and brier score, lower is better (use reverse colormap)
                cmap = 'RdYlBu' if metric in ['Log Loss', 'Brier Score'] else 'RdYlGn'
                
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, 
                           ax=axes[i], cbar_kws={'label': metric})
                axes[i].set_title(f'{metric} by Role and Model')
                axes[i].set_xlabel('Model')
                axes[i].set_ylabel('Role')
        
        # Remove empty subplot
        if len(metrics_to_plot) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/metrics_comparison_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_summary_table(self) -> pd.DataFrame:
        """Create summary table of best performing models per role"""
        summary_data = []
        
        for role, role_results in self.evaluation_results.items():
            # Find best model based on AUC-ROC
            best_model = max(role_results.keys(), key=lambda x: role_results[x]['auc_roc'])
            best_metrics = role_results[best_model]
            
            summary_data.append({
                'Role': role,
                'Best Model': best_model.replace('_', ' ').title(),
                'Accuracy': best_metrics['accuracy'],
                'F1 Score': best_metrics['f1_score'],
                'AUC-ROC': best_metrics['auc_roc'],
                'Log Loss': best_metrics['log_loss'],
                'Brier Score': best_metrics['brier_score']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{self.results_dir}/performance_summary.csv', index=False)
        
        return summary_df
    
    def generate_classification_reports(self, role_models: Dict[str, Any], 
                                      X_test: np.ndarray, y_test: np.ndarray, role: str) -> None:
        """Generate detailed classification reports for each model"""
        reports_dir = f'{self.results_dir}/classification_reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        for model_name, model in role_models.items():
            y_pred, y_proba = self.evaluate_model_predictions(model, X_test, y_test, model_name)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(f'{reports_dir}/{role.lower()}_{model_name}_classification_report.csv')
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {role} {model_name.replace("_", " ").title()}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'{reports_dir}/{role.lower()}_{model_name}_confusion_matrix.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_comprehensive_evaluation(self, trainer_results: Dict[str, Dict]) -> None:
        """Run complete evaluation pipeline"""
        print("Starting comprehensive model evaluation...")
        
        for role, role_data in trainer_results.items():
            print(f"\nEvaluating {role} models...")
            
            models = role_data['models']
            X_test, y_test = role_data['test_data']
            scaler = role_data['scaler']
            
            # Scale test data for models that need it
            X_test_scaled = scaler.transform(X_test)
            
            # Evaluate models with appropriate data
            role_results = {}
            for model_name, model in models.items():
                if model_name in ['logistic', 'neural_network']:
                    X_eval = X_test_scaled
                else:
                    X_eval = X_test
                
                y_pred, y_proba = self.evaluate_model_predictions(model, X_eval, y_test, model_name)
                metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_proba)
                role_results[model_name] = metrics
            
            self.evaluation_results[role] = role_results
            
            # Generate visualizations
            self.plot_roc_curves(models, X_test_scaled, y_test, role)
            self.plot_calibration_curves(models, X_test_scaled, y_test, role)
            self.generate_classification_reports(models, X_test_scaled, y_test, role)
        
        # Create comparison plots and summary
        self.create_metrics_comparison_plot()
        summary_df = self.create_performance_summary_table()
        
        print("\nEvaluation completed!")
        print("\nPerformance Summary:")
        print(summary_df.to_string(index=False))
        
        return self.evaluation_results


def main():
    """Example usage of ModelEvaluator"""
    from model_training import ModelTrainer
    from data_preprocessing import DataProcessor
    
    # Load data and train models (abbreviated for example)
    processor = DataProcessor()
    df, role_datasets = processor.process_full_pipeline()
    
    trainer = ModelTrainer()
    trainer_results = {}
    
    # Train models for each role (simplified)
    for role, role_data in role_datasets.items():
        if len(role_data) >= 100:
            feature_names = processor.get_feature_names()
            X = role_data[feature_names]
            y = role_data['Win']
            trainer_results[role] = trainer.train_role_models(X, y, role)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.run_comprehensive_evaluation(trainer_results)


if __name__ == "__main__":
    main()