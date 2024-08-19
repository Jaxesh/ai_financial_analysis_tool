# src/prediction.py

import joblib
import pandas as pd

def load_model(filepath):
    return joblib.load(filepath)

def predict(model, X):
    predictions = model.predict(X)
    return predictions
