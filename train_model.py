import joblib

from data_fetch import fetch_stock_data
from indicators import add_indicators
from model import train_model

print("Fetching stock data...")

# Use a reliable stock for training
df = fetch_stock_data("RELIANCE", period="2y")

print("Generating indicators...")
df = add_indicators(df)

print("Training model...")
model, scaler, accuracy, features = train_model(df)

# Save model and scaler
joblib.dump(model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")
print(f"Accuracy: {accuracy:.2%}")