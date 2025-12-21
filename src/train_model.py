"""
Train Titanic survival prediction model.

This script trains a model and saves it to the models/ directory.
Usage: python src/train_model.py
"""

import os
import sys

# Add parent directory to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

from utils import load_data, preprocess_data, prepare_features, prepare_target


def train_and_evaluate(X_train, X_val, y_train, y_val, model, model_name):
    """Train model and print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print('='*50)
    
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Metrics
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    train_f1 = f1_score(y_train, train_pred)
    val_f1 = f1_score(y_val, val_pred)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f}")
    print(f"Train F1:       {train_f1:.4f}")
    print(f"Val F1:         {val_f1:.4f}")
    
    return model, val_acc, val_f1


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'train.csv')
    model_path = os.path.join(project_root, 'models', 'titanic_model.pkl')
    
    # Create models directory if not exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print("Titanic Survival Prediction - Model Training")
    print("="*50)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Preprocess
    print("\nPreprocessing data...")
    df_processed = preprocess_data(df)
    
    # Prepare features and target
    X = prepare_features(df_processed)
    y = prepare_target(df_processed)
    print(f"Feature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model, lr_val_acc, lr_val_f1 = train_and_evaluate(
        X_train, X_val, y_train, y_val, lr_model, "Logistic Regression"
    )
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        min_samples_split=5,
        random_state=42
    )
    rf_model, rf_val_acc, rf_val_f1 = train_and_evaluate(
        X_train, X_val, y_train, y_val, rf_model, "Random Forest"
    )
    
    # Select best model
    print(f"\n{'='*50}")
    print("Model Comparison")
    print('='*50)
    print(f"Logistic Regression - Val Acc: {lr_val_acc:.4f}, Val F1: {lr_val_f1:.4f}")
    print(f"Random Forest       - Val Acc: {rf_val_acc:.4f}, Val F1: {rf_val_f1:.4f}")
    
    if rf_val_acc >= lr_val_acc:
        best_model = rf_model
        best_name = "Random Forest"
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
    
    print(f"\nBest Model: {best_name}")
    
    # Retrain on full data
    print(f"\nRetraining {best_name} on full training data...")
    best_model.fit(X, y)
    
    # Cross-validation on full data
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Save model
    print(f"\nSaving model to {model_path}...")
    joblib.dump(best_model, model_path)
    print("Model saved successfully!")
    
    # Verify saved model
    loaded_model = joblib.load(model_path)
    sample_pred = loaded_model.predict(X.iloc[[0]])
    print(f"\nModel verification - Sample prediction: {sample_pred[0]}")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == "__main__":
    main()
