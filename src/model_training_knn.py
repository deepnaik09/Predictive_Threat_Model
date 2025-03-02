import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_knn_model(input_path="data/featured_data.csv", model_output="models/knn_model.pkl"):
    """Trains a K-Nearest Neighbors (KNN) model for loan default prediction and saves it."""

    print("📂 Loading feature-engineered dataset...")
    df = pd.read_csv(input_path)

    # Ensure dataset does not contain categorical data
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

    # Apply SMOTE to balance dataset
    print("🔄 Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Feature scaling (Standardization for better performance)
    scaler = StandardScaler()
    X_resampled = scaler.fit_transform(X_resampled)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train KNN model
    print("🛠 Training K-Nearest Neighbors (KNN) Classifier...")
    model = KNeighborsClassifier(n_neighbors=5)  # Using k=5 as default
    model.fit(X_train, y_train)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_output)
    print(f"✅ KNN model trained and saved at {model_output}")

    # Evaluate model performance
    print("📊 Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"🎯 Model Accuracy: {accuracy:.2f}")
    print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))
    print("\n📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    train_knn_model()
