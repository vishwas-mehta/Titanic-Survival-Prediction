"""
Make predictions using the trained Titanic model.

This script loads the trained model and makes predictions on new data.
Includes a demo function to test predictions on sample passengers.

Usage: python src/predict.py
"""

import os
import sys
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import preprocess_data_advanced, prepare_features_advanced


def load_model():
    """Load the trained model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, 'models', 'titanic_advanced_model.pkl')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, 'models', 'titanic_model.pkl')
    
    return joblib.load(model_path)


def predict_survival(passenger_data: dict) -> dict:
    """
    Predict survival for a single passenger.
    
    Args:
        passenger_data: Dictionary with passenger features:
            - Pclass: Ticket class (1, 2, or 3)
            - Name: Full name (for title extraction)
            - Sex: 'male' or 'female'
            - Age: Age in years
            - SibSp: Number of siblings/spouses aboard
            - Parch: Number of parents/children aboard
            - Fare: Ticket fare
            - Cabin: Cabin number (optional)
            - Embarked: Port of embarkation (C, Q, or S)
    
    Returns:
        Dictionary with prediction and probability
    """
    # Create DataFrame from input
    df = pd.DataFrame([passenger_data])
    
    # Add required columns if missing
    if 'PassengerId' not in df.columns:
        df['PassengerId'] = 1
    if 'Survived' not in df.columns:
        df['Survived'] = 0  # Dummy value for preprocessing
    if 'Ticket' not in df.columns:
        df['Ticket'] = 'NA'
    if 'Cabin' not in df.columns:
        df['Cabin'] = None
    
    # Preprocess
    df_processed = preprocess_data_advanced(df)
    X = prepare_features_advanced(df_processed)
    
    # Predict
    model = load_model()
    prediction = model.predict(X)[0]
    
    return {
        'survived': bool(prediction),
        'prediction': 'Survived' if prediction == 1 else 'Did not survive'
    }


def demo_predictions():
    """Run demo predictions on sample passengers."""
    print("Titanic Survival Prediction - Demo")
    print("=" * 50)
    
    # Sample passengers
    passengers = [
        {
            'Name': 'Mr. John Smith',
            'Pclass': 3, 'Sex': 'male', 'Age': 25,
            'SibSp': 0, 'Parch': 0, 'Fare': 7.25, 'Embarked': 'S'
        },
        {
            'Name': 'Mrs. Jane Doe',
            'Pclass': 1, 'Sex': 'female', 'Age': 35,
            'SibSp': 1, 'Parch': 0, 'Fare': 83.5, 'Embarked': 'C'
        },
        {
            'Name': 'Master. Tommy Brown',
            'Pclass': 2, 'Sex': 'male', 'Age': 8,
            'SibSp': 0, 'Parch': 2, 'Fare': 26.0, 'Embarked': 'S'
        },
        {
            'Name': 'Miss Sarah Wilson',
            'Pclass': 1, 'Sex': 'female', 'Age': 22,
            'SibSp': 0, 'Parch': 0, 'Fare': 151.55, 'Embarked': 'S'
        }
    ]
    
    for i, passenger in enumerate(passengers, 1):
        result = predict_survival(passenger)
        print(f"\nPassenger {i}: {passenger['Name']}")
        print(f"  Class: {passenger['Pclass']}, Sex: {passenger['Sex']}, Age: {passenger['Age']}")
        print(f"  Prediction: {result['prediction']}")
    
    print("\n" + "=" * 50)
    print("Demo complete!")


if __name__ == "__main__":
    demo_predictions()
