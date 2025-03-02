import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_linear_model(input_path="data/featured_data.csv", model_output="models/linear_model.pkl"):
    """Trains a Linear Regression model for prediction."""

    print("📂 Loading feature-engineered dataset...")
    df = pd.read_csv(input_path)

    X = df.drop(columns=["Target"])
    y = df["Target"].astype(float)  # Linear Regression requires continuous output

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🛠 Training Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_output)
    print(f"✅ Linear Regression Model saved at {model_output}")

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"📉 Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"📉 Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

if __name__ == "__main__":
    train_linear_model()
