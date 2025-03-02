import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path="data/generated_data.csv", output_path="data/processed_data.csv"):
    print("📂 Loading dataset for preprocessing...")
    df = pd.read_csv(input_path)
    print("🔍 Columns in dataset before preprocessing:", df.columns.tolist())

    # Standardize column names (Ensure consistency)
    df.rename(columns={
        "LoanAmount": "Loan_Amount",
        "CreditScore": "Credit_Score",
        "AccountBalance": "Account_Balance",
        "MonthsEmployed": "Months_Employed",
        "NumCreditLines": "Num_Credit_Lines",
        "InterestRate": "Interest_Rate",
        "DTIRatio": "DTI_Ratio"
    }, inplace=True)

    # Identify numerical and categorical columns dynamically
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Ensure LoanId and Target column are not processed
    if "LoanId" in numerical_cols:
        numerical_cols.remove("LoanId")

    target_col = "Default"
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    # Handle missing values
    num_imputer = SimpleImputer(strategy="median")
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # Process categorical columns
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Apply scaling only to numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Ensure "Default" column is retained
    print("🔍 Columns after preprocessing:", df.columns.tolist())

    # Save the processed data
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data()
