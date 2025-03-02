import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(input_path="data/featured_data.csv", model_output="models/random_forest_model.pkl"):
    """Trains a predictive model using RandomForest and saves it."""

    print("📂 Loading feature-engineered dataset...")
    df = pd.read_csv(input_path)


    # Ensure categorical columns are removed
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        print(f"⚠️ Dropping non-numeric columns: {categorical_cols}")
        df.drop(columns=categorical_cols, inplace=True, errors="ignore")

    # Ensure target column exists
    if "Target" not in df.columns:
        raise ValueError("❌ 'Target' column is missing! Make sure your dataset is properly processed.")

    # Separate features and target
    X = df.drop(columns=["Target"])
    y = df["Target"].astype(int)

    # Handle class imbalance using SMOTE
    print("🔄 Applying SMOTE to balance classes...")
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except ValueError as e:
        raise ValueError(f"⚠️ SMOTE error: {e}. Ensure numeric data and no missing values.") 

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Train the RandomForest model
    print("🛠 Training RandomForest Classifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_output)
    print(f"✅ Model trained and saved at {model_output}")

    # Evaluate the model
    print("📊 Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"🎯 Model Accuracy: {accuracy:.2f}")
    print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))
    print("\n📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature importance ranking
    feature_importances = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    print("\n🔥 Top 5 Important Features:\n", feature_importances.head())

if __name__ == "__main__":
    train_model()
