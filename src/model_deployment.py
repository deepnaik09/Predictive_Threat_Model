from flask import Flask, request, jsonify
import joblib
import pandas as pd

#Load model
model = joblib.load("models/trained_model.pkl")

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "🏦 Loan Default Prediction API! Use /predict to make predictions."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Ensure correct feature order
        expected_features = ["Age", "Income", "Loan_Amount", "Credit_Score", 
                             "Account_Balance", "Interest_Rate", "Loan_to_Income_Ratio", 
                             "Loan_to_Balance_Ratio", "High_Credit_Risk"]
        df = df[expected_features]

        # Make prediction
        prediction = model.predict(df)

        return jsonify({"Default_Prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
