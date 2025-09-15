"""
Model Loading Utilities for League of Legends Player Skill Rating
Convenient functions to load trained models and make predictions
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from tensorflow import keras


class ModelLoader:
    """Utility class for loading and using trained models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.feature_names = [
            'KDE', 'DPM', 'Multi-Kill', 'GPM', 'VSPM', 'WCPM',
            'GD@15', 'XPD@15', 'CSD@15', 'LVLD@15', 'DTPD'
        ]
        self.roles = ['top', 'jungle', 'mid', 'adc', 'support']
        self.model_types = ['logistic', 'random_forest', 'xgboost', 'neural_network']
    
    def load_model(self, role: str, model_type: str) -> Any:
        """Load a specific model"""
        role_lower = role.lower()
        
        if model_type == 'neural_network':
            model_path = f"{self.models_dir}/best_{role_lower}_model.h5"
            if os.path.exists(model_path):
                return keras.models.load_model(model_path)
        else:
            model_path = f"{self.models_dir}/best_{role_lower}_model_{model_type}.pkl"
            if os.path.exists(model_path):
                return joblib.load(model_path)
        
        raise FileNotFoundError(f"Model not found: {role} {model_type}")
    
    def load_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Load all available models"""
        all_models = {}
        
        for role in self.roles:
            role_models = {}
            for model_type in self.model_types:
                try:
                    model = self.load_model(role, model_type)
                    role_models[model_type] = model
                    print(f"Loaded {role} {model_type} model")
                except FileNotFoundError:
                    print(f"Model not found: {role} {model_type}")
                    continue
            
            if role_models:
                all_models[role] = role_models
        
        self.loaded_models = all_models
        return all_models
    
    def get_best_model_per_role(self) -> Dict[str, Tuple[str, Any]]:
        """Get the best performing model for each role based on saved results"""
        best_models = {}
        
        # Load performance results to determine best models
        results_dir = "results"
        
        for role in self.roles:
            role_cap = role.capitalize()
            metrics_file = f"{results_dir}/metrics_{role_cap}.csv"
            
            if os.path.exists(metrics_file):
                try:
                    metrics_df = pd.read_csv(metrics_file, index_col=0)
                    
                    # Find best model by AUC-ROC (assuming this column exists)
                    if 'auc_roc' in metrics_df.columns:
                        best_model_type = metrics_df['auc_roc'].idxmax()
                    elif 'AUC-ROC' in metrics_df.columns:
                        best_model_type = metrics_df['AUC-ROC'].idxmax()
                    else:
                        # Fallback to accuracy
                        best_model_type = metrics_df.iloc[:, 0].idxmax()
                    
                    # Load the best model
                    model = self.load_model(role, best_model_type)
                    best_models[role] = (best_model_type, model)
                    
                except Exception as e:
                    print(f"Error loading best model for {role}: {e}")
                    # Fallback to XGBoost if available
                    try:
                        model = self.load_model(role, 'xgboost')
                        best_models[role] = ('xgboost', model)
                    except:
                        continue
        
        return best_models
    
    def predict_player_skill(self, player_stats: Dict[str, float], role: str,
                           model_type: Optional[str] = None) -> Dict[str, float]:
        """Predict player skill rating using specified or best model"""
        role_lower = role.lower()
        
        # Prepare features
        feature_values = [player_stats.get(feature, 0) for feature in self.feature_names]
        X = np.array(feature_values).reshape(1, -1)
        
        # Load model
        if model_type is None:
            # Use best model for role
            best_models = self.get_best_model_per_role()
            if role_lower not in best_models:
                raise ValueError(f"No model available for role: {role}")
            model_type, model = best_models[role_lower]
        else:
            model = self.load_model(role, model_type)
        
        # Make prediction
        if model_type == 'neural_network':
            win_probability = float(model.predict(X)[0][0])
            win_prediction = int(win_probability > 0.5)
        else:
            win_prediction = int(model.predict(X)[0])
            win_probability = float(model.predict_proba(X)[0][1])
        
        return {
            'role': role,
            'model_used': model_type,
            'win_prediction': win_prediction,
            'win_probability': win_probability,
            'skill_rating': win_probability  # Use probability as skill rating
        }
    
    def batch_predict(self, players_df: pd.DataFrame, role_column: str = 'Role') -> pd.DataFrame:
        """Make predictions for multiple players"""
        results = []
        
        for idx, row in players_df.iterrows():
            role = row[role_column].lower()
            
            # Extract player stats
            player_stats = {feature: row.get(feature, 0) for feature in self.feature_names}
            
            try:
                prediction = self.predict_player_skill(player_stats, role)
                prediction['player_id'] = idx
                results.append(prediction)
            except Exception as e:
                print(f"Error predicting for player {idx}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def get_model_info(self) -> pd.DataFrame:
        """Get information about all available models"""
        model_info = []
        
        for role in self.roles:
            for model_type in self.model_types:
                model_path = (f"{self.models_dir}/best_{role}_model.h5" if model_type == 'neural_network'
                            else f"{self.models_dir}/best_{role}_model_{model_type}.pkl")
                
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                    model_info.append({
                        'role': role,
                        'model_type': model_type,
                        'file_path': model_path,
                        'file_size_mb': round(file_size, 2),
                        'available': True
                    })
                else:
                    model_info.append({
                        'role': role,
                        'model_type': model_type,
                        'file_path': model_path,
                        'file_size_mb': 0,
                        'available': False
                    })
        
        return pd.DataFrame(model_info)


def quick_predict(player_stats: Dict[str, float], role: str) -> float:
    """Quick prediction function using best available model"""
    loader = ModelLoader()
    result = loader.predict_player_skill(player_stats, role)
    return result['skill_rating']


def load_best_models() -> Dict[str, Tuple[str, Any]]:
    """Convenience function to load best models for all roles"""
    loader = ModelLoader()
    return loader.get_best_model_per_role()


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = ModelLoader()
    
    # Show available models
    print("Available Models:")
    print(loader.get_model_info())
    
    # Example prediction
    example_stats = {
        'KDE': 2.5,
        'DPM': 450,
        'Multi-Kill': 1,
        'GPM': 380,
        'VSPM': 1.2,
        'WCPM': 0.8,
        'GD@15': 200,
        'XPD@15': 150,
        'CSD@15': 10,
        'LVLD@15': 0,
        'DTPD': 800
    }
    
    prediction = loader.predict_player_skill(example_stats, 'top')
    print(f"\nExample Prediction: {prediction}")