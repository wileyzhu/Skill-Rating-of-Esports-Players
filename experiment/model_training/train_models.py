import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow import keras

def build_model(n_hidden=1, n_neurons=30, learning_rate=0.01, input_shape=None, optimizer='adam', dropout_rate=0.0):
    if input_shape is None:
        raise ValueError("input_shape must be provided")
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate) if optimizer == 'adam' else keras.optimizers.SGD(learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

def train_all_models(X_train, y_train, role_prefix, random_state):
    input_shape = X_train.shape[1:]
    keras_model = KerasClassifier(model=build_model, input_shape=input_shape, verbose=0, random_state=random_state)
    param_grid_nn = {
        'epochs': [50, 100],
        'batch_size': [32, 64],
        'model__learning_rate': [0.001, 0.01],
        'model__n_hidden': [1, 2],
        'model__n_neurons': np.arange(10, 100, 10),
        'model__dropout_rate': [0.0, 0.2, 0.3],
        'model__optimizer': ['adam', 'sgd'],
    }

    rnd_nn = RandomizedSearchCV(keras_model, param_grid_nn, random_state=random_state, n_iter=10, cv=3, verbose=1)
    rnd_nn.fit(X_train, y_train)
    rnd_nn.best_estimator_.model_.save(f"best_{role_prefix}_model.h5")

    # Logistic
    param_grid_logistic = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': np.logspace(-4, 4, 10),
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'max_iter': [100, 1000]
    }
    rnd_lr = RandomizedSearchCV(LogisticRegression(random_state=random_state), param_grid_logistic, n_iter=10, cv=3, random_state=random_state, verbose=1)
    rnd_lr.fit(X_train, y_train)
    joblib.dump(rnd_lr, f"best_{role_prefix}_model_logistic.pkl")

    # RF
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    rnd_rf = RandomizedSearchCV(RandomForestClassifier(random_state=random_state), param_grid_rf, n_iter=10, cv=3, random_state=random_state, verbose=1)
    rnd_rf.fit(X_train, y_train)
    joblib.dump(rnd_rf, f"best_{role_prefix}_model_random_forest.pkl")

    # XGBoost
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1]
    }
    rnd_xgb = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state), param_grid_xgb, n_iter=10, cv=3, random_state=random_state, verbose=1)
    rnd_xgb.fit(X_train, y_train)
    joblib.dump(rnd_xgb, f"best_{role_prefix}_model_xgboost.pkl")

    return {
        "Neural Network": rnd_nn,
        "Logistic Regression": rnd_lr,
        "Random Forest": rnd_rf,
        "XGBoost": rnd_xgb
    }