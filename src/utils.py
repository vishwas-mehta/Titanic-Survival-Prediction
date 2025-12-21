"""
Utility functions for Titanic data preprocessing and feature engineering.
"""

import pandas as pd
import numpy as np


def load_data(filepath):
    """Load CSV data from filepath."""
    return pd.read_csv(filepath)


def fill_missing_age(df):
    """
    Fill missing Age values with median age grouped by Pclass.
    Passengers in higher classes tend to be older.
    """
    df = df.copy()
    age_by_class = df.groupby('Pclass')['Age'].transform('median')
    df['Age'] = df['Age'].fillna(age_by_class)
    # If still any NaN (edge case), fill with overall median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    return df


def encode_categorical(df):
    """
    Encode categorical variables:
    - Sex: male=0, female=1
    - Embarked: S=0, C=1, Q=2
    """
    df = df.copy()
    
    # Encode Sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Fill missing Embarked with mode (S is most common)
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    return df


def create_family_size(df):
    """
    Create FamilySize feature from SibSp and Parch.
    FamilySize = SibSp + Parch + 1 (including self)
    """
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df


def fill_missing_fare(df):
    """Fill missing Fare with median fare by Pclass."""
    df = df.copy()
    fare_by_class = df.groupby('Pclass')['Fare'].transform('median')
    df['Fare'] = df['Fare'].fillna(fare_by_class)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    return df


def get_feature_columns():
    """Return list of feature columns used for training."""
    return ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']


def preprocess_data(df):
    """
    Complete preprocessing pipeline.
    Returns preprocessed dataframe.
    """
    df = fill_missing_age(df)
    df = fill_missing_fare(df)
    df = encode_categorical(df)
    df = create_family_size(df)
    return df


def prepare_features(df):
    """
    Prepare feature matrix X from preprocessed dataframe.
    """
    feature_cols = get_feature_columns()
    return df[feature_cols]


def prepare_target(df):
    """
    Prepare target vector y from dataframe.
    """
    return df['Survived']
