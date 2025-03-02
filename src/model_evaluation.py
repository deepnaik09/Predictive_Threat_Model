import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model_path="models/trained_model.pkl", data_path="data/featured_data.csv"):
    """Loads the trained model and evaluates it."""

    df = pd.read_csv(data_path)

    #Check the actual target column name
    target_column = "Default"  # Change this if your target column is named differently

    # Ensure the target column exists
    if target_column not in df.columns:
        raise ValueError(f"❌ Target column '{target_column}' not found in dataset!")

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    # Load model
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X)

    # Evaluation
    print(f"✅ Model Evaluation Completed!")
    print(f"📊 Accuracy: {accuracy_score(y, y_pred):.2f}")
    print("📌 Classification Report:\n", classification_report(y, y_pred))
    print("📌 Confusion Matrix:\n", confusion_matrix(y, y_pred))

if __name__ == "__main__":
    evaluate_model()
