# Credit Card Fraud Detection

## Project Overview

This project builds a Machine Learning system to detect fraudulent credit card transactions.

The model learns patterns from past transactions and predicts whether a new transaction is fraud or legitimate.

Technologies used:
- Python
- Pandas
- Scikit-Learn
- XGBoost
- Streamlit

------------------------------------------------------------

## Dataset

Dataset: Kaggle Credit Card Fraud Detection Dataset

Total Transactions: 284,807  
Fraud Transactions: 492  
Fraud Percentage: 0.172% (Highly Imbalanced)

------------------------------------------------------------

## Features Explanation

The dataset contains 30 input features:

1. V1 – V28  
   - These are PCA (Principal Component Analysis) features.
   - Original sensitive banking features were transformed into numerical components for privacy.
   - They represent hidden behavioral patterns like spending habits, location signals, merchant interactions, etc.

2. Time  
   - Time difference between this transaction and the first transaction.
   - Helps detect unusual timing behavior.

3. Amount  
   - Transaction value.

4. Class (Target Variable)  
   - 0 = Legitimate transaction  
   - 1 = Fraud transaction  

------------------------------------------------------------

## Problem

The dataset is highly imbalanced (very few fraud cases).

If we train directly, the model may predict everything as normal.

To solve this:
- SMOTE is used to balance the dataset.

------------------------------------------------------------

## Fraud Detection Approach

1. Preprocessing
   - Scale Amount and Time
   - Separate features (X) and target (y)

2. Handle Imbalance
   - Apply SMOTE to create balanced dataset

3. Anomaly Detection
   - Isolation Forest
   - Local Outlier Factor (LOF)

4. Supervised Model
   - XGBoost classifier
   - Learns fraud vs normal patterns

------------------------------------------------------------

## Model Evaluation

1. Confusion Matrix
   - True Positive
   - True Negative
   - False Positive
   - False Negative

2. ROC Curve
   - Measures model performance
   - AUC close to 1 indicates strong performance

------------------------------------------------------------

## How Fraud Is Detected

The system does NOT simply check if the amount is high.

It analyzes:
- Behavioral deviations
- Statistical anomalies
- Combination of multiple PCA features

Fraud detection is based on pattern recognition.

------------------------------------------------------------

## Project Workflow

1. Load dataset
2. Preprocess data
3. Balance dataset using SMOTE
4. Train Isolation Forest & LOF
5. Train XGBoost model
6. Evaluate using confusion matrix & ROC curve
7. Save trained model
8. Deploy using Streamlit

------------------------------------------------------------

## How To Run

### Step 1: Go to project folder

```
cd fraud-detection-project
```

### Step 2: Activate virtual environment (Windows)

```
venv\Scripts\activate
```

### Step 3: Train model

```
python notebook.py
```

### Step 4: Run web app

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

------------------------------------------------------------

## Project Structure

```
fraud-detection-project/
│
├── creditcard.csv
├── notebook.py
├── app.py
├── fraud_model.pkl
├── requirements.txt
└── README.md
```
