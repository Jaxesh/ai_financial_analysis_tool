# main.py

from src.data_preprocessing import load_data, preprocess_data, split_data
from src.feature_engineering import create_features
from src.model_training import train_model, evaluate_model, save_model
from src.prediction import load_model, predict
from src.utils import plot_predictions

# Load and preprocess data
data = load_data('data/market_data.csv')
data = preprocess_data(data)

# Feature engineering
data = create_features(data)

# Split the data
X_train, X_test, y_train, y_test = split_data(data)

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Save the model
save_model(model, 'models/model.pkl')

# Load the model and make predictions
loaded_model = load_model('models/model.pkl')
predictions = predict(loaded_model, X_test)

# Plot predictions
plot_predictions(y_test, predictions)
