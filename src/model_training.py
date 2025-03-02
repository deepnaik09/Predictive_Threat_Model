import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(input_path="data/featured_data.csv", model_output="models/trained_model.pkl"):
    """Trains a predictive model and saves it."""

    print("📂 Loading feature-engineered dataset...")
    df = pd.read_csv(input_path)

    #Ensure no categorical columns remain
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        print(f"⚠️ Dropping non-numeric columns: {categorical_cols}")
        df.drop(columns=categorical_cols, inplace=True, errors="ignore")

    # Ensure target column exists
    if "Default" not in df.columns:
        raise ValueError("❌ 'Default' column is missing! Make sure your dataset is properly processed.")

    # Separate features and target
    X = df.drop(columns=["Default"])
    y = df["Default"].astype(int)

    # Apply SMOTE for class balancing
    print("🔄 Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train RandomForest model
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

if __name__ == "__main__":
    train_model()
