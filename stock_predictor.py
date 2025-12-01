import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# -----------------------
# Download Stock Data
# -----------------------
stock_symbol = "TATAMOTORS.NS"

df = yf.download(stock_symbol, start="2015-01-01", end="2025-01-01")
print("Dataset Loaded Successfully!")

# -----------------------
# Preprocessing
# -----------------------
df = df[['Close']]
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# -----------------------
# Train Data
# -----------------------
X = df[['Close']].values.reshape(-1,1)   # ENSURE 2D array
y = df['Target'].values.reshape(-1,1)    # ENSURE 2D array

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert targets to 1D for sklearn
y_train = y_train.ravel()
y_test = y_test.ravel()

# -----------------------
# Train Model
# -----------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------
# Evaluate Model
# -----------------------
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)

print("Model Training Completed!")
print("Mean Squared Error (MSE):", mse)

# -----------------------
# Predict Next Day Price (IMPORTANT FIX)
# -----------------------
latest_price = float(df['Close'].iloc[-1])   # extract pure float
future_price = model.predict(np.array([[latest_price]]))  # correct 2D input

print("\n---------------------------------------")
print(f"Predicted Next Day Price for {stock_symbol}: ₹{future_price[0]:.2f}")
print("---------------------------------------")

# -----------------------
# Save Model
# -----------------------
joblib.dump(model, "stock_price_model.pkl")
print("\nModel saved as: stock_price_model.pkl")
