# **Predictive Threat Modeling for Public Sector Banks**

## **Project Overview**
This project builds a **predictive threat modeling system** using **machine learning and AI techniques**. The model predicts whether a bank customer is likely to **default on a loan** based on various financial and demographic factors.

## **📌 Project Workflow**
The system follows a structured **machine learning pipeline**:

1. **Data Generation** – Create realistic synthetic banking data.
2. **Preprocessing** – Handle missing values, encode categorical variables, and scale numerical features.
3. **Feature Engineering** – Generate new meaningful features to improve prediction accuracy.
4. **Model Training** – Train a **Random Forest model** with **SMOTE** for class balancing.
5. **Model Evaluation** – Assess model performance using accuracy, confusion matrix, and classification reports.
6. **Model Deployment** – Deploy the model using **Flask** for real-time loan default predictions.

---

## **🚀 How to Run the Project**

### **Step 1️⃣: Install Dependencies**
Before running the project, install all required Python libraries:
```bash
pip install -r requirements.txt
```

---

### **Step 2️⃣: Run the ML Pipeline**
You can run **each step separately** or execute the **entire pipeline**.

#### **Run the Complete Pipeline**
```bash
python main.py
```
This runs **all steps** in sequence.

#### **Run Each Step Individually**
```bash
python main.py --step generate      # If you have real time data(named as generated_data in data dir) then skip this command It Generate synthetic data
python main.py --step preprocess    # Preprocess data
python main.py --step feature       # Perform feature engineering
python main.py --step train         # Train the model
python main.py --step evaluate      # Evaluate the model
python main.py --step deploy        # Deploy the model as an API
```

---

## **📊 Explanation of Each Step**

### **🔹 Step 1: Data Generation**
- **File:** `data_generation.py`
- **Description:**
  - Generates a synthetic dataset using `sklearn.datasets.make_classification()`.
  - Features include **credit score, loan amount, interest rate, employment status, etc.**
  - Saves the dataset as `data/generated_data.csv`.

---

### **🔹 Step 2: Data Preprocessing**
- **File:** `data_preprocessing.py`
- **Description:**
  - Handles **missing values**:
    - **Numerical Features** → Median imputation.
    - **Categorical Features** → Most frequent category.
  - **Encodes categorical variables** using **Label Encoding**.
  - **Scales numerical data** using **StandardScaler**.
  - Saves the cleaned data as `data/processed_data.csv`.

---

### **🔹 Step 3: Feature Engineering**
- **File:** `feature_engineering.py`
- **Description:**
  - Creates **new meaningful features**:
    - **Loan-to-Balance Ratio** → `Loan Amount / Account Balance`.
    - **High-Risk Credit Flag** → Identifies customers with low credit scores.
    - **Age Grouping** → Categorizes customers into age brackets.
  - Saves the enhanced dataset as `data/featured_data.csv`.

---

### **🔹 Step 4: Model Training**
- **File:** `model_training.py`
- **Description:**
  - Loads the **engineered dataset**.
  - **Balances the dataset** using **SMOTE**.
  - Splits data into **training (80%) and testing (20%)** sets.
  - Trains a **Random Forest Classifier** with:
    - `n_estimators=200` (200 decision trees).
    - `max_depth=15` (Limits depth to prevent overfitting).
  - Saves the trained model to `models/trained_model.pkl`.

**✅ Training Accuracy: ~97%**

---

### **🔹 Step 5: Model Evaluation**
- **File:** `model_evaluation.py`
- **Description:**
  - Loads the **trained model**.
  - Makes **predictions** on the test dataset.
  - Calculates evaluation metrics:
    - **Accuracy Score**
    - **Confusion Matrix**
    - **Classification Report**
  - Displays the **evaluation results**.

**✅ Evaluation Accuracy: ~99%**

---

### **🔹 Step 6: Model Deployment (Optional)**
- **File:** `model_deployment.py`
- **Description:**
  - Creates a **Flask API** for real-time loan default predictions.
  - Users send **loan details** as JSON.
  - API returns whether the loan **will default (1) or not (0)**.

#### **Run the API**
```bash
python model_deployment.py
```

#### **Make a Sample Prediction**
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"Credit_Score": 650, "Loan_Amount": 50000, "Account_Balance": 200000, "Interest_Rate": 5.5, "Loan_to_Balance_Ratio": 0.25, "High_Credit_Risk": 0}'
```

**Expected Output:**
```json
{"Default_Prediction": 0}
```

---

## **📊 Exploratory Data Analysis (EDA)**
- **File:** `notebooks/EDA.ipynb`
- **Description:**
  - Loads the dataset and displays basic statistics.
  - Visualizes data distribution using:
    - **Histograms** 📊
    - **Boxplots** 📦
    - **Correlation Heatmaps** 🔥
  - Helps understand **data trends and relationships**.

---

## **💡 Key Improvements for High Accuracy**
✅ **Data Balancing** – Used **SMOTE** to handle class imbalance.
✅ **Feature Engineering** – Added meaningful features for better decision-making.
✅ **Optimized Hyperparameters** – Tuned the Random Forest model for **better generalization**.
✅ **Robust Preprocessing** – StandardScaler ensures consistent feature scaling.
✅ **Thorough Evaluation** – Included classification reports and confusion matrix analysis.

---

## **📌 Final Thoughts**
✅ **Achieved 97% Training Accuracy & 99% Evaluation Accuracy**
✅ **Successfully Predicts Loan Defaults with High Precision**
✅ **Well-structured ML Pipeline for Banking Risk Analysis**

---


This document serves as a comprehensive guide to the **Predictive Threat Modeling** project. 🚀


│── data/
│   ├── featured_data.csv
│   ├── generated_data.csv
│   ├── processed_data.csv
│── src/
│   ├── data_generation.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── model_deployment.py (Optional)
|   ├── model_prediction.py
│── models/
│   ├── trained_model.pkl
│── my_dataset
│   │── input_dataset.csv
│   │── predicted_result.csv
│── notebooks/
│   ├── EDA.ipynb
│── requirements.txt
│── README.md
│── main.py

