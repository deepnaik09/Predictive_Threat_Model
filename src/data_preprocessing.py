import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path="data/generated_data.csv", output_path="data/processed_data.csv"):
    """
    Preprocesses the dataset by handling missing values, encoding categorical variables, 
    and normalizing numeric features while keeping readability.
    """
    #Load dataset
    df = pd.read_csv(input_path)

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Identify numerical and categorical columns
    numerical_cols = ["Age", "Income", "LoanAmount", "CreditScore", "AccountBalance",
                      "MonthsEmployed", "NumCreditLines", "InterestRate", "DTIRatio"]
    categorical_cols = ["LoanTerm"]  # Loan term is discrete, should be treated as categorical
    target_col = "Default"

    # Handle missing values
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Encode categorical variables
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Apply scaling only to numerical features (excluding LoanId and Target)
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Save the processed data
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data()
