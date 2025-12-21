# Titanic Survival Prediction

A beginner-friendly machine learning project that predicts Titanic passenger survival using Python (Pandas, Seaborn, Scikit-learn). Includes EDA, feature engineering, model training, and evaluation.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview

This project uses the famous [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic) to predict whether a passenger survived the Titanic disaster based on features like age, sex, passenger class, and fare.

### Key Features

- **Exploratory Data Analysis (EDA)**: Visualizations for survival patterns by sex, class, age, and more
- **Feature Engineering**: Missing value handling, categorical encoding, family size feature
- **Model Training**: Logistic Regression and Random Forest comparison
- **Model Evaluation**: Accuracy, F1 score, confusion matrices, and cross-validation

## Project Structure

```
Titanic_Prediction/
├── data/
│   ├── train.csv          # Training dataset (891 passengers)
│   └── test.csv           # Test dataset (418 passengers)
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory Data Analysis
│   └── 02_modeling.ipynb  # Model training & evaluation
├── src/
│   ├── utils.py           # Data preprocessing utilities
│   └── train_model.py     # Training script
├── models/
│   └── titanic_model.pkl  # Saved best model
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vishwas-mehta/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/` and open:
- `01_eda.ipynb` - For exploratory data analysis and visualizations
- `02_modeling.ipynb` - For model training and evaluation

### Train Model via Script

```bash
python src/train_model.py
```

This will:
1. Load and preprocess the training data
2. Train Logistic Regression and Random Forest models
3. Compare performance and select the best model
4. Save the best model to `models/titanic_model.pkl`

### Use the Trained Model

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('models/titanic_model.pkl')

# Prepare features (example)
# Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize
sample = [[3, 0, 25, 0, 0, 7.25, 0, 1]]  # Male, 3rd class, 25 years old

# Predict
prediction = model.predict(sample)
print(f"Survived: {'Yes' if prediction[0] == 1 else 'No'}")
```

## Model Performance

| Model | Validation Accuracy | Validation F1 | CV Accuracy |
|-------|---------------------|---------------|-------------|
| Logistic Regression | 81.01% | 73.85% | ~80% |
| Random Forest | **82.12%** | 73.33% | **82.05%** |

**Best Model**: Random Forest was selected based on validation accuracy.

### Feature Importance

The most important features for prediction (from Random Forest):
1. **Sex** - Women had much higher survival rates
2. **Fare** - Higher fare correlates with survival
3. **Age** - Younger passengers survived more often
4. **Pclass** - First-class passengers had better chances

## Dataset Information

The dataset contains information about 891 passengers from the Titanic:

| Feature | Description |
|---------|-------------|
| PassengerId | Unique ID |
| Survived | Survival (0 = No, 1 = Yes) |
| Pclass | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Name | Passenger name |
| Sex | Gender |
| Age | Age in years |
| SibSp | # of siblings/spouses aboard |
| Parch | # of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

## Key Insights from EDA

1. **Overall Survival Rate**: ~38% of passengers survived
2. **Gender Effect**: Women had ~74% survival rate vs ~19% for men
3. **Class Effect**: 1st class: ~63%, 2nd class: ~47%, 3rd class: ~24%
4. **Family Size**: Medium-sized families (2-4) had better survival rates
5. **Age**: Children had higher survival rates

## Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **Seaborn & Matplotlib** - Data visualization
- **Jupyter** - Interactive notebooks
- **Joblib** - Model serialization

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Vishwas Mehta**
- GitHub: [@vishwas-mehta](https://github.com/vishwas-mehta)

---

⭐ If you found this project helpful, please consider giving it a star!
