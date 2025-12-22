"""
Utility functions for Titanic data preprocessing and feature engineering.
Includes both basic and advanced feature engineering methods.
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


# ============================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================

def extract_title(df):
    """
    Extract title from passenger Name.
    Titles are grouped into: Mr, Mrs, Miss, Master, Rare
    This is a strong predictor - titles indicate gender, age, and social status.
    """
    df = df.copy()
    
    # Extract title using regex
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mr': 0,
        'Miss': 1,
        'Mrs': 2,
        'Master': 3,
        'Dr': 4,
        'Rev': 4,
        'Col': 4,
        'Major': 4,
        'Mlle': 1,  # French Miss
        'Countess': 4,
        'Ms': 1,
        'Lady': 4,
        'Jonkheer': 4,
        'Don': 4,
        'Dona': 4,
        'Mme': 2,  # French Mrs
        'Capt': 4,
        'Sir': 4
    }
    
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(4)  # Unknown titles -> Rare
    
    return df


def create_is_alone(df):
    """
    Create IsAlone feature - passengers traveling alone had different survival rates.
    """
    df = df.copy()
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df


def create_fare_per_person(df):
    """
    Create FarePerPerson - divide fare by family size.
    Some fares are for whole families, so this normalizes the fare.
    """
    df = df.copy()
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    return df


def create_has_cabin(df):
    """
    Create HasCabin feature - having cabin info correlates with higher class.
    """
    df = df.copy()
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    return df


def bin_age(df):
    """
    Bin age into categories: Child (0-12), Teen (13-19), Adult (20-60), Senior (60+)
    """
    df = df.copy()
    bins = [0, 12, 19, 60, 100]
    labels = [0, 1, 2, 3]  # Child, Teen, Adult, Senior
    df['AgeBin'] = pd.cut(df['Age'], bins=bins, labels=labels)
    df['AgeBin'] = df['AgeBin'].astype(int)
    return df


def create_age_class(df):
    """
    Create Age*Class interaction feature.
    Young first class passengers likely had best survival chances.
    """
    df = df.copy()
    df['Age_Class'] = df['Age'] * df['Pclass']
    return df


def get_feature_columns():
    """Return list of basic feature columns used for training."""
    return ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']


def get_advanced_feature_columns():
    """Return list of all feature columns including advanced features."""
    return [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
        'FamilySize', 'Title', 'IsAlone', 'FarePerPerson', 'HasCabin', 
        'AgeBin', 'Age_Class'
    ]


def preprocess_data(df):
    """
    Basic preprocessing pipeline.
    Returns preprocessed dataframe.
    """
    df = fill_missing_age(df)
    df = fill_missing_fare(df)
    df = encode_categorical(df)
    df = create_family_size(df)
    return df


def preprocess_data_advanced(df):
    """
    Advanced preprocessing pipeline with enhanced feature engineering.
    Returns preprocessed dataframe with all advanced features.
    """
    # First apply basic preprocessing
    df = fill_missing_age(df)
    df = fill_missing_fare(df)
    
    # Advanced features (before encoding)
    df = extract_title(df)
    df = create_has_cabin(df)
    
    # Basic encoding
    df = encode_categorical(df)
    df = create_family_size(df)
    
    # More advanced features (after family size is created)
    df = create_is_alone(df)
    df = create_fare_per_person(df)
    df = bin_age(df)
    df = create_age_class(df)
    
    return df


def prepare_features(df):
    """
    Prepare basic feature matrix X from preprocessed dataframe.
    """
    feature_cols = get_feature_columns()
    return df[feature_cols]


def prepare_features_advanced(df):
    """
    Prepare advanced feature matrix X from preprocessed dataframe.
    """
    feature_cols = get_advanced_feature_columns()
    return df[feature_cols]


def prepare_target(df):
    """
    Prepare target vector y from dataframe.
    """
    return df['Survived']
