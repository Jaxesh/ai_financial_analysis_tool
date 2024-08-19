# src/feature_engineering.py

def create_features(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    # Drop any rows with NaN values created by rolling functions
    data = data.dropna()
    
    return data
