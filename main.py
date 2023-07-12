import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and model fitting
pipe = Pipeline([
    ('preprocessing', StandardScaler()), 
    ('classifier', SVC())
])

# Define the models and their respective hyperparameters
models = [
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'rbf']
    },
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [10, 50, 100, 200],
        'classifier__max_features': [1, 2, 3]
    },
    {
        'classifier': [LogisticRegression()],
        'classifier__C': [0.1, 1, 10, 100]
    }
]

# Apply GridSearchCV to find the best model and hyperparameters
grid = GridSearchCV(pipe, models, cv=5, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# Print the best model and its score
print(f"Best parameters: {grid.best_params_}")
print(f"Validation accuracy: {grid.best_score_}")

# Evaluate the model on the test set
y_pred = grid.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred)}")
