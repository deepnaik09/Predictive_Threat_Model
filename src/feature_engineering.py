import pandas as pd
import os

def feature_engineering(input_path="data/processed_data.csv", output_path="data/featured_data.csv"):
    """Applies feature engineering and saves the transformed dataset."""

    # Ensure the input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"🚨 Input dataset not found at: {input_path}")

    # Load dataset
    print("📂 Loading preprocessed dataset for feature engineering...")
    df = pd.read_csv(input_path)
    print("🔍 Columns before feature engineering:", df.columns.tolist())

    # Ensure required columns exist
    required_columns = ["Loan_Amount", "Income", "Credit_Score", "DTI_Ratio", "Account_Balance"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"❌ Missing required columns: {missing_columns}")

    # Feature: Loan to Income Ratio
    df["Loan_to_Income_Ratio"] = df["Loan_Amount"] / (df["Income"] + 1)  # Avoid division by zero
    df["Loan_to_Income_Ratio"] = df["Loan_to_Income_Ratio"].clip(lower=0)  # Remove negatives

    # Feature: Loan to Balance Ratio
    df["Loan_to_Balance_Ratio"] = df["Loan_Amount"] / (df["Account_Balance"] + 1)  # Avoid division by zero
    df["Loan_to_Balance_Ratio"] = df["Loan_to_Balance_Ratio"].clip(lower=0)  # Remove negatives

    # Feature: High Credit Risk Flag
    df["High_Credit_Risk"] = (df["Credit_Score"] < 650).astype(int)

    # Ensure target column is retained
    if "Target" not in df.columns:
        raise ValueError("❌ 'Target' column is missing after feature engineering!")

    # Ensure LoanId remains unchanged
    if "LoanId" in df.columns:
        df["LoanId"] = df["LoanId"].astype(int)

    # Save the updated dataset
    os.makedirs("data", exist_ok=True)  # Ensure directory exists
    df.to_csv(output_path, index=False)

    print("🔍 Columns after feature engineering:", df.columns.tolist())
    print(f"✅ Feature Engineering Completed! 📂 Saved at: {output_path}")

if __name__ == "__main__":
    feature_engineering()
