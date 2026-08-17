import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# Load Dataset

df = pd.read_csv("house_price_regression_dataset.csv")

print("First 5 Records")
print(df.head())


# Features and Target

X = df[[
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality"
]]

y = df["House_Price"]


# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train Model

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)


# Evaluation

print("\nModel Performance")
print("---------------------------")
print("R2 Score :", round(r2_score(y_test, y_pred), 4))
print("MAE      :", round(mae := mean_absolute_error(y_test, y_pred), 2))
print("RMSE     :", round((mean_squared_error(y_test, y_pred)) ** 0.5, 2))


# Save Model

joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")