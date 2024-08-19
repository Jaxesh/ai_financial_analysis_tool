# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Convert the 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Sort data by date
    data = data.sort_values('Date')
    
    # Example of creating additional features
    data['Price_Change'] = data['Close'] - data['Open']
    data['Volatility'] = data['High'] - data['Low']
    
    # Drop any rows with missing values
    data = data.dropna()
    
    return data

def split_data(data):
    X = data[['Open', 'High', 'Low', 'Volume', 'Price_Change', 'Volatility']]
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
