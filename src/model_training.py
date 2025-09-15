"""
Model Training Module for League of Legends Player Skill Rating
Implements multiple ML models with hyperparameter tuning for each role
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow import keras


class ModelTrainer:
    """Trains and evaluates multiple ML models for player skill rating prediction"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _build_neural_network(self, n_hidden: int = 1, n_neurons: int = 30, 
                             learning_rate: float = 0.01, input_shape: Tuple = None,
                             optimizer: str = 'adam', dropout_rate: float = 0.0) -> keras.Model:
        """Build neural network architecture"""
        if input_shape is None:
            raise ValueError("input_shape must be provided")
        
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        
        # Add hidden layers
        for _ in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation="relu"))
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        
        # Compile model
        opt = (keras.optimizers.Adam(learning_rate) if optimizer == 'adam' 
               else keras.optimizers.SGD(learning_rate))
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        
        return model
    
    def _get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for each model type"""
        return {
            'logistic': {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': np.logspace(-4, 4, 10),
                'solver': ['lbfgs', 'liblinear', 'saga'],
                'max_iter': [100, 1000]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0, 0.1]
            },
            'neural_network': {
                'epochs': [50, 100],
                'batch_size': [32, 64],
                'model__learning_rate': [0.001, 0.01],
                'model__n_hidden': [1, 2],
                'model__n_neurons': np.arange(10, 100, 10),
                'model__dropout_rate': [0.0, 0.2, 0.3],
                'model__optimizer': ['adam', 'sgd'],
            }
        }
    
    def train_role_models(self, X: pd.DataFrame, y: pd.Series, role: str) -> Dict[str, Any]:
        """Train all model types for a specific role"""
        print(f"\nTraining models for {role} role...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[role] = scaler
        
        param_grids = self._get_hyperparameter_grids()
        role_models = {}
        role_results = {}
        
        # 1. Logistic Regression
        print(f"Training Logistic Regression for {role}...")
        lr_search = RandomizedSearchCV(
            LogisticRegression(random_state=self.random_state),
            param_grids['logistic'],
            n_iter=20, cv=3, random_state=self.random_state, 
            scoring='roc_auc', n_jobs=-1
        )
        lr_search.fit(X_train_scaled, y_train)
        role_models['logistic'] = lr_search.best_estimator_
        
        # 2. Random Forest
        print(f"Training Random Forest for {role}...")
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=self.random_state),
            param_grids['random_forest'],
            n_iter=20, cv=3, random_state=self.random_state,
            scoring='roc_auc', n_jobs=-1
        )
        rf_search.fit(X_train, y_train)  # RF doesn't need scaling
        role_models['random_forest'] = rf_search.best_estimator_
        
        # 3. XGBoost
        print(f"Training XGBoost for {role}...")
        xgb_search = RandomizedSearchCV(
            XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
            param_grids['xgboost'],
            n_iter=20, cv=3, random_state=self.random_state,
            scoring='roc_auc', n_jobs=-1
        )
        xgb_search.fit(X_train, y_train)
        role_models['xgboost'] = xgb_search.best_estimator_
        
        # 4. Neural Network
        print(f"Training Neural Network for {role}...")
        input_shape = (X_train_scaled.shape[1],)
        keras_model = KerasClassifier(
            model=self._build_neural_network,
            input_shape=input_shape,
            verbose=0,
            random_state=self.random_state
        )
        
        nn_search = RandomizedSearchCV(
            keras_model,
            param_grids['neural_network'],
            n_iter=10, cv=3, random_state=self.random_state,
            scoring='roc_auc'
        )
        nn_search.fit(X_train_scaled, y_train)
        role_models['neural_network'] = nn_search.best_estimator_
        
        # Evaluate all models
        for model_name, model in role_models.items():
            if model_name == 'neural_network':
                y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
                y_proba = model.predict(X_test_scaled).flatten()
                X_eval = X_test_scaled
            elif model_name in ['logistic']:
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                X_eval = X_test_scaled
            else:  # RF, XGB
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                X_eval = X_test
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            role_results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'auc_roc': auc,
                'best_params': (lr_search.best_params_ if model_name == 'logistic' else
                               rf_search.best_params_ if model_name == 'random_forest' else
                               xgb_search.best_params_ if model_name == 'xgboost' else
                               nn_search.best_params_)
            }
            
            print(f"{model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
        
        # Store results
        self.models[role] = role_models
        self.results[role] = role_results
        
        return {
            'models': role_models,
            'results': role_results,
            'test_data': (X_test, y_test),
            'scaler': scaler
        }
    
    def save_models(self, output_dir: str = "models"):
        """Save all trained models to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        for role, role_models in self.models.items():
            role_lower = role.lower()
            
            # Save traditional ML models
            for model_name, model in role_models.items():
                if model_name == 'neural_network':
                    # Save Keras model
                    model.model_.save(f"{output_dir}/best_{role_lower}_model.h5")
                else:
                    # Save sklearn models
                    joblib.dump(model, f"{output_dir}/best_{role_lower}_model_{model_name}.pkl")
            
            # Save scaler
            joblib.dump(self.scalers[role], f"{output_dir}/{role_lower}_scaler.pkl")
        
        print(f"All models saved to {output_dir}/")
    
    def save_results(self, output_dir: str = "results"):
        """Save training results and metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        for role, role_results in self.results.items():
            results_df = pd.DataFrame(role_results).T
            results_df.to_csv(f"{output_dir}/metrics_{role}.csv")
        
        print(f"Results saved to {output_dir}/")
    
    def get_best_model_per_role(self) -> Dict[str, Tuple[str, Any]]:
        """Return the best performing model for each role based on AUC-ROC"""
        best_models = {}
        
        for role, role_results in self.results.items():
            best_model_name = max(role_results.keys(), 
                                key=lambda x: role_results[x]['auc_roc'])
            best_model = self.models[role][best_model_name]
            best_models[role] = (best_model_name, best_model)
        
        return best_models


def main():
    """Example usage of ModelTrainer"""
    from data_preprocessing import DataProcessor
    
    # Load and preprocess data
    processor = DataProcessor()
    df, role_datasets = processor.process_full_pipeline()
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Train models for each role
    for role, role_data in role_datasets.items():
        if len(role_data) < 100:  # Skip roles with insufficient data
            print(f"Skipping {role} due to insufficient data ({len(role_data)} samples)")
            continue
        
        # Prepare features and target
        feature_names = processor.get_feature_names()
        X = role_data[feature_names]
        y = role_data['Win']
        
        # Train models
        trainer.train_role_models(X, y, role)
    
    # Save models and results
    trainer.save_models()
    trainer.save_results()
    
    # Display best models
    best_models = trainer.get_best_model_per_role()
    print("\nBest models per role:")
    for role, (model_name, model) in best_models.items():
        auc = trainer.results[role][model_name]['auc_roc']
        print(f"{role}: {model_name} (AUC: {auc:.3f})")


if __name__ == "__main__":
    main()