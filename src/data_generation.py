import os
import pandas as pd
import numpy as np

def generate_data(n_samples=5000, random_state=42):
    """
    Generates realistic synthetic loan-related data following Indian financial standards.
    """
    np.random.seed(random_state)

    #Generate synthetic banking data
    data = {
        "LoanId": np.arange(1, n_samples + 1),  # Unique Loan IDs
        "Age": np.random.randint(21, 65, size=n_samples),  # Age between 21 and 65
        "Income": np.random.randint(200000, 3000000, size=n_samples),  # Annual Income in INR
        "Loan_Amount": np.random.randint(50000, 2500000, size=n_samples),  # Loan amount in INR
        "Credit_Score": np.random.randint(300, 900, size=n_samples),  # Credit Score (300-900)
        "Account_Balance": np.random.randint(5000, 10000000, size=n_samples),  # Bank Balance in INR
        "Months_Employed": np.random.randint(0, 480, size=n_samples),  # Employment Duration in Months
        "Num_Credit_Lines": np.random.randint(1, 10, size=n_samples),  # Number of Credit Lines
        "Interest_Rate": np.round(np.random.uniform(5.0, 15.0, size=n_samples), 2),  # Interest Rate (5%-15%)
        "Loan_Term": np.random.choice([12, 24, 36, 48, 60, 120, 180, 240, 300, 360], size=n_samples),  # Loan Tenure in months
        "DTI_Ratio": np.round(np.random.uniform(0.1, 0.5, size=n_samples), 2),  # Debt-to-Income Ratio (10%-50%)
        "Target": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # Loan Default: 0 (No), 1 (Yes) with 70/30 ratio
    }

    df = pd.DataFrame(data)

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Save dataset
    file_path = os.path.join("data", "generated_data.csv")
    df.to_csv(file_path, index=False)
    print(f"✅ Synthetic Data Generated Successfully! 📂 Saved at: {file_path}")

if __name__ == "__main__":
    generate_data()
