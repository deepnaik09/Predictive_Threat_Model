import pandas as pd
import os

def feature_engineering(input_path="data/processed_data.csv", output_path="data/featured_data.csv"):
    """Applies feature engineering and saves the transformed dataset."""

    # Ensure the file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"🚨 Input dataset not found at: {input_path}")

    #Load dataset
    df = pd.read_csv(input_path)

    # Ensure required columns exist
    required_columns = ["LoanAmount", "Income", "CreditScore", "DTIRatio"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"❌ Missing required columns: {missing_columns}")

    # Feature: Loan to Income Ratio
    df["Loan_to_Income_Ratio"] = df["LoanAmount"] / (df["Income"] + 1)  # Avoid division by zero

    # Feature: Loan to Balance Ratio
    df["Loan_to_Balance_Ratio"] = df["LoanAmount"] / (df["AccountBalance"] + 1)  # Avoid division by zero

    # Feature: High Credit Risk Flag
    df["High_Credit_Risk"] = (df["CreditScore"] < 650).astype(int)

    # Save the updated dataset
    os.makedirs("data", exist_ok=True)  # Ensure directory exists
    df.to_csv(output_path, index=False)

    print(f"✅ Feature Engineering Completed! 📂 Saved at: {output_path}")

if __name__ == "__main__":
    feature_engineering()
