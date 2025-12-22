"""
Advanced Model Training for Titanic Survival Prediction.

This script trains multiple advanced models with hyperparameter tuning
and compares their performance. Models include:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine
- K-Nearest Neighbors
- Voting Ensemble
- Stacking Ensemble

Usage: python src/advanced_train.py
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed, skipping XGBoost models")

from utils import (
    load_data, preprocess_data_advanced, 
    prepare_features_advanced, prepare_target
)


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """Train model and return evaluation metrics."""
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    
    return {
        'model': model,
        'name': model_name,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_f1': train_f1,
        'val_f1': val_f1
    }


def tune_model(model, param_grid, X_train, y_train, model_name):
    """Perform hyperparameter tuning using GridSearchCV."""
    print(f"  Tuning {model_name}...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"    Best params: {grid_search.best_params_}")
    print(f"    Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'train.csv')
    model_path = os.path.join(project_root, 'models', 'titanic_model.pkl')
    advanced_model_path = os.path.join(project_root, 'models', 'titanic_advanced_model.pkl')
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print_header("TITANIC SURVIVAL PREDICTION - ADVANCED TRAINING")
    
    # Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    df = load_data(data_path)
    print(f"     Loaded {len(df)} samples")
    
    df_processed = preprocess_data_advanced(df)
    X = prepare_features_advanced(df_processed)
    y = prepare_target(df_processed)
    print(f"     Feature matrix: {X.shape} (14 advanced features)")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"     Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # =========================================================
    # Train Multiple Models
    # =========================================================
    print_header("[2/6] TRAINING BASE MODELS")
    
    results = []
    
    # 1. Logistic Regression
    print("\n  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    results.append(evaluate_model(lr, X_train_scaled, X_val_scaled, y_train, y_val, "Logistic Regression"))
    
    # 2. Random Forest
    print("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    results.append(evaluate_model(rf, X_train, X_val, y_train, y_val, "Random Forest"))
    
    # 3. Gradient Boosting
    print("  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    results.append(evaluate_model(gb, X_train, X_val, y_train, y_val, "Gradient Boosting"))
    
    # 4. XGBoost
    if HAS_XGBOOST:
        print("  Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        results.append(evaluate_model(xgb, X_train, X_val, y_train, y_val, "XGBoost"))
    
    # 5. SVM
    print("  Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    results.append(evaluate_model(svm, X_train_scaled, X_val_scaled, y_train, y_val, "SVM (RBF)"))
    
    # 6. KNN
    print("  Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    results.append(evaluate_model(knn, X_train_scaled, X_val_scaled, y_train, y_val, "KNN"))
    
    # =========================================================
    # Hyperparameter Tuning for Top Models
    # =========================================================
    print_header("[3/6] HYPERPARAMETER TUNING")
    
    # Tune Random Forest
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    best_rf = tune_model(RandomForestClassifier(random_state=42), rf_params, X_train, y_train, "Random Forest")
    
    # Tune Gradient Boosting
    gb_params = {
        'n_estimators': [100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1]
    }
    best_gb = tune_model(GradientBoostingClassifier(random_state=42), gb_params, X_train, y_train, "Gradient Boosting")
    
    # Tune XGBoost
    if HAS_XGBOOST:
        xgb_params = {
            'n_estimators': [100, 150],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1]
        }
        best_xgb = tune_model(
            XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            xgb_params, X_train, y_train, "XGBoost"
        )
    
    # =========================================================
    # Evaluate Tuned Models
    # =========================================================
    print_header("[4/6] EVALUATING TUNED MODELS")
    
    tuned_results = []
    
    result = evaluate_model(best_rf, X_train, X_val, y_train, y_val, "Random Forest (Tuned)")
    tuned_results.append(result)
    print(f"  Random Forest (Tuned):     Val Acc: {result['val_acc']:.4f}, Val F1: {result['val_f1']:.4f}")
    
    result = evaluate_model(best_gb, X_train, X_val, y_train, y_val, "Gradient Boosting (Tuned)")
    tuned_results.append(result)
    print(f"  Gradient Boosting (Tuned): Val Acc: {result['val_acc']:.4f}, Val F1: {result['val_f1']:.4f}")
    
    if HAS_XGBOOST:
        result = evaluate_model(best_xgb, X_train, X_val, y_train, y_val, "XGBoost (Tuned)")
        tuned_results.append(result)
        print(f"  XGBoost (Tuned):           Val Acc: {result['val_acc']:.4f}, Val F1: {result['val_f1']:.4f}")
    
    # =========================================================
    # Ensemble Methods
    # =========================================================
    print_header("[5/6] ENSEMBLE METHODS")
    
    # Voting Classifier
    print("\n  Building Voting Ensemble...")
    estimators = [
        ('rf', best_rf),
        ('gb', best_gb),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ]
    if HAS_XGBOOST:
        estimators.append(('xgb', best_xgb))
    
    voting = VotingClassifier(estimators=estimators, voting='hard')
    result = evaluate_model(voting, X_train, X_val, y_train, y_val, "Voting Ensemble")
    tuned_results.append(result)
    print(f"  Voting Ensemble:           Val Acc: {result['val_acc']:.4f}, Val F1: {result['val_f1']:.4f}")
    
    # Stacking Classifier
    print("  Building Stacking Ensemble...")
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    result = evaluate_model(stacking, X_train, X_val, y_train, y_val, "Stacking Ensemble")
    tuned_results.append(result)
    print(f"  Stacking Ensemble:         Val Acc: {result['val_acc']:.4f}, Val F1: {result['val_f1']:.4f}")
    
    # =========================================================
    # Results Summary
    # =========================================================
    print_header("[6/6] FINAL RESULTS")
    
    print("\n  BASE MODELS:")
    print("-" * 55)
    print(f"  {'Model':<25} {'Val Acc':>10} {'Val F1':>10}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: x['val_acc'], reverse=True):
        print(f"  {r['name']:<25} {r['val_acc']:>10.4f} {r['val_f1']:>10.4f}")
    
    print("\n  TUNED/ENSEMBLE MODELS:")
    print("-" * 55)
    for r in sorted(tuned_results, key=lambda x: x['val_acc'], reverse=True):
        print(f"  {r['name']:<25} {r['val_acc']:>10.4f} {r['val_f1']:>10.4f}")
    
    # Get best model overall
    all_results = results + tuned_results
    best_result = max(all_results, key=lambda x: x['val_acc'])
    
    print(f"\n  BEST MODEL: {best_result['name']}")
    print(f"  Validation Accuracy: {best_result['val_acc']:.4f}")
    print(f"  Validation F1 Score: {best_result['val_f1']:.4f}")
    
    # Cross-validation on best model
    print("\n  Running 5-fold cross-validation on best model...")
    best_model = best_result['model']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Retrain on full data
    print("\n  Retraining best model on full data...")
    best_model.fit(X, y)
    
    # Save model
    print(f"\n  Saving best model to {advanced_model_path}...")
    joblib.dump(best_model, advanced_model_path)
    print("  Model saved successfully!")
    
    # Also update the main model
    joblib.dump(best_model, model_path)
    print(f"  Updated main model at {model_path}")
    
    print_header("TRAINING COMPLETE!")
    print(f"\n  Best Model: {best_result['name']}")
    print(f"  Validation Accuracy: {best_result['val_acc']:.4f}")
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f}")
    print("\n")


if __name__ == "__main__":
    main()
