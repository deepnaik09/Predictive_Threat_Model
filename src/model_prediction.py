import os
import joblib
import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

# Define paths
MODEL_PATH = "models/trained_model.pkl"
DATASET_DIR = "my_dataset"  # Directory where dataset is stored

# Ensure the trained model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Model file not found at: {MODEL_PATH}")

#Load trained model
model = joblib.load(MODEL_PATH)
print(f"✅ Loaded model from {MODEL_PATH}")

# Define expected features
expected_features = [
    "Age", "Income", "Credit_Score", "Loan_Amount", 
    "Account_Balance", "Interest_Rate", "Loan_to_Balance_Ratio",
    "High_Credit_Risk"
]

@app.route("/", methods=["GET"])
def home():
    return "🏦 Welcome to the Loan Default Prediction API! This will read a dataset from 'my_dataset/' and save predictions in a new file."

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Find the dataset file in my_dataset/
        files = os.listdir(DATASET_DIR)
        data_file = None
        for file in files:
            if file.endswith(".csv") or file.endswith(".xlsx"):
                data_file = file
                break  # Pick the first file found
        
        if not data_file:
            return jsonify({"error": "No dataset file found in 'my_dataset/'"}), 400

        file_path = os.path.join(DATASET_DIR, data_file)

        # Read the dataset
        if data_file.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Ensure required features exist
        missing_features = [feature for feature in expected_features if feature not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Make predictions
        predictions = model.predict(df)

        # Create a new DataFrame for predictions
        df_predictions = df.copy()
        df_predictions["Loan_Defaulter"] = predictions

        # Save predictions in a new file inside my_dataset/
        new_file_name = f"predicted_{data_file}"
        new_file_path = os.path.join(DATASET_DIR, new_file_name)

        if data_file.endswith(".csv"):
            df_predictions.to_csv(new_file_path, index=False)
        else:
            df_predictions.to_excel(new_file_path, index=False)

        return jsonify({
            "message": f"Predictions saved in {new_file_path}",
            "file": new_file_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
