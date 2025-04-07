'''This file contains all functions that will be used in the model_analysis.ipuynb notebook.
The code folloows the Single Responsibility Principle (SRP) and is organized in a modular way.'''

# ---------------------- Imports --------------------
import os
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -------------------- 1. Data Handling --------------------
def load_dataset(path: str):
    '''Load a dataset from a given path.'''
    df = pd.read_csv(path)
    return df


# -------------------- 2. Data Splitting --------------------
def split_features_target(df, target_col):
    '''Split the dataframe into features and target.'''
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# -------------------- 3. Training --------------------
def train_model(X_train, y_train, model):
    '''Fit a model on the training data.'''
    model.fit(X_train, y_train)
    return model


# -------------------- 4. Prediction --------------------
def predict(model, X_test):
    '''Make predictions using the trained model.'''
    y_pred = model.predict(X_test)
    return y_pred


# -------------------- 5. Metrics Computation --------------------
def compute_metrics(y_true, y_pred):
    '''Calculate RMSE, MAE and R2 score.'''
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


# -------------------- 6. Model Evaluation --------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    '''Evaluate the model using RMSE, MAE and R2 score.'''
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse, train_mae, train_r2 = compute_metrics(y_train, y_train_pred)
    test_rmse, test_mae, test_r2 = compute_metrics(y_test, y_test_pred)
    
    metrics = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }
    
    return metrics

def bootstrap_evaluation(model, X, y, n_iterations=100, random_state=42):
    '''Perform bootstrap evaluation on the evaluation set.'''
    np.random.seed(random_state)
    rmse_scores, mae_scores, r2_scores = [], [], []

    for _ in range(n_iterations):
        indices = np.random.choice(len(X), len(X), replace=True)
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices]
        y_pred = model.predict(X_sample)
        rmse, mae, r2 = compute_metrics(y_sample, y_pred)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    summary = {
        'RMSE': {
            'mean': np.mean(rmse_scores),
            'median': np.median(rmse_scores),
            'ci_95': [
                np.percentile(rmse_scores, 2.5),
                np.percentile(rmse_scores, 97.5)
            ]
        },
        'MAE': {
            'mean': np.mean(mae_scores),
            'median': np.median(mae_scores),
            'ci_95': [
                np.percentile(mae_scores, 2.5),
                np.percentile(mae_scores, 97.5)
            ]
        },
        'R2': {
            'mean': np.mean(r2_scores),
            'median': np.median(r2_scores),
            'ci_95': [
                np.percentile(r2_scores, 2.5),
                np.percentile(r2_scores, 97.5)
            ]
        }
    }

    raw_scores = {
        'RMSE': rmse_scores,
        'MAE': mae_scores,
        'R2': r2_scores
    }
    return summary, raw_scores


def bootstrap_dataframe(summaries):
    df_dict = {}
    for model_name, metrics in summaries.items():
        df_metrics = {}
        for metric, stats in metrics.items():
            df_metrics[f"{metric}_mean"] = stats["mean"]
            df_metrics[f"{metric}_median"] = stats["median"]
            df_metrics[f"{metric}_ci_low"] = stats["ci_95"][0]
            df_metrics[f"{metric}_ci_high"] = stats["ci_95"][1]
        df_dict[model_name] = df_metrics
    return pd.DataFrame.from_dict(df_dict, orient="index")


# -------------------- 7. Feature Selection --------------------
def select_features_sfs(estimator, X, y, forward=True, k_features=(10,50), scoring='neg_root_mean_squared_error', cv_splits=5):
    '''Select optimal number of features using SFS.'''
    sfs = SFS(estimator,
              k_features=k_features,
              forward=forward,
              floating=False,
              scoring=scoring,
              cv=cv_splits,
              n_jobs=1,
              verbose=1)
    sfs.fit(X.values, y.values)
    selected_fetures = list(X.columns[list(sfs.k_feature_idx_)])
    X_selected = X[selected_fetures]
    return X_selected, selected_fetures



def filter_fs(X, y, k_features='default'):
    '''Select top k features using SelectKBest.'''

    if k_features == 'default':
        k_features = int(0.8 * X.shape[1])  # keep 80% of features

    selector = SelectKBest(score_func=f_regression, k=k_features)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    X_new = pd.DataFrame(X_new, columns=selected_features, index=X.index)
    return X_new, selected_features


# -------------------- 8. Tuning --------------------
def tune_model(model, X_train, y_train, param_grid, scoring='neg_root_mean_squared_error', cv_splits=5):
    '''Tune model hyperparameters using GridSearchCV.'''

    print(f"\n Tuning: {model.__class__.__name__}")
    print(f" Grid size: {sum(len(v) for v in param_grid.values())} total params")
    print(f" Dataset shape: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv)
    grid_search.fit(X_train, y_train)

    print(f" Best parameters: {grid_search.best_params_}")
    print(f" Best CV score: {-grid_search.best_score_:.4f}")  # RMSE is negative

    best_model = grid_search.best_estimator_
    return best_model


# -------------------- 9. Plotting --------------------
def plot_metrics_boxplot(metrics_dict, save_path=None):
    '''Create boxplots comparing models across RMSE, MAE and R2 score.'''
    metric_names = ['RMSE', 'MAE', 'R2']
    model_names = list(metrics_dict.keys())
    labels = []

    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(8,5))
        data = [metrics_dict[model_name][metric] for model_name in model_names]

        ax.boxplot(data, labels=model_names)
        ax.set_title(f'{metric} Distribution Across Models')
        ax.set_ylabel(metric)
        ax.grid(True)
        plt.tight_layout()
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


# -------------------- 10. Model Saving --------------------
def save_model(model, path: str):
    '''Save the trained model to a file.'''
    joblib.dump(model, path)

def load_model(path: str):
    '''Load a model from a file.'''
    model = joblib.load(path)
    return model

# -------------------- 11. BONUS 1 --------------------
def suggest_hyperparameters(trial, param_space):
    """
    Suggest hyperparameters from a given parameter space using an Optuna trial.
    Format for param_space:
        - float range: (min, max)
        - log-uniform: (min, max, 'log')
        - categorical: ['option1', 'option2']
    """
    suggested = {}
    for param, values in param_space.items():
        if isinstance(values, tuple) and len(values) == 3 and values[2] == 'log':
            suggested[param] = trial.suggest_float(param, values[0], values[1], log=True)
        elif isinstance(values, tuple):
            suggested[param] = trial.suggest_float(param, values[0], values[1])
        elif isinstance(values, list):
            suggested[param] = trial.suggest_categorical(param, values)
        else:
            raise ValueError(f"Unsupported format for parameter '{param}': {values}")
    return suggested

def optuna_objective(trial, model, X, y, param_space, scoring, cv_splits, random_state):
    """
    Objective function to be minimized by Optuna.
    """
    params = suggest_hyperparameters(trial, param_space)
    model_instance = clone(model).set_params(**params)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    score = cross_val_score(model_instance, X, y, scoring=scoring, cv=cv).mean()
    return -score

def tune_model_optuna(model, X, y, param_space, scoring='neg_root_mean_squared_error',
                      n_trials=50, cv_splits=5, random_state=42):
    """
    Tune model hyperparameters using Optuna with trial pruning and Bayesian optimization.
    
    Returns:
        - best_model: trained model with best parameters
        - best_params: dict of best parameters
        - best_score: best (lowest) RMSE score found
    """
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

    func = lambda trial: optuna_objective(trial, model, X, y, param_space, scoring, cv_splits, random_state)
    study.optimize(func, n_trials=n_trials)

    best_params = study.best_params
    best_model = clone(model).set_params(**best_params)
    best_model.fit(X, y)

    return best_model, best_params, study.best_value


# -------------------- 12. BONUS 2 --------------------
def compute_classification_metrics(y_true, y_pred):
    '''Calculate classification metrics: Accuracy, Precision, Recall, F1.'''
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1

def evaluate_classification_model(model, X_train, y_train, X_test, y_test):
    '''Evaluate classifier using accuracy, precision, recall and f1-score.'''
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc, train_prec, train_rec, train_f1 = compute_classification_metrics(y_train, y_train_pred)
    test_acc, test_prec, test_rec, test_f1 = compute_classification_metrics(y_test, y_test_pred)
    
    metrics = {
        'train_accuracy': train_acc,
        'train_precision': train_prec,
        'train_recall': train_rec,
        'train_f1': train_f1,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1
    }
    
    return metrics


