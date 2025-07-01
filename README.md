# 🏥 Healthcare Provider Fraud Detection

This project builds a machine learning solution to identify potentially fraudulent healthcare providers using insurance claims and beneficiary data. Fraudulent billing practices inflate healthcare costs—our model helps insurance companies proactively detect suspicious providers and reduce financial loss.

---

## 📂 Dataset Overview

The project uses four interlinked CSV datasets:
- **Train.csv**: Labels providers as `PotentialFraud` (`Yes`/`No`)
- **Train_Beneficiarydata.csv**: Demographic and health condition info (age, gender, chronic conditions)
- **Train_Inpatientdata.csv** & **Train_Outpatientdata.csv**: Claim info including procedures, diagnosis, physician IDs, and reimbursement amounts

---

## ⚙️ Project Pipeline

1. **Data Preprocessing**:
   - Handled missing values
   - Date parsing and transformation
   - Merged all datasets on provider and beneficiary IDs

2. **Feature Engineering**:
   - Created claim-level and provider-level statistical features
   - Aggregated claims per provider
   - Generated binary flags, durations, and ratios

3. **Modeling**:
   - Addressed class imbalance using **SMOTE**
   - Models used: **Logistic Regression**, **XGBoost**
   - Hyperparameter tuning via **RandomizedSearchCV**
   - Evaluation metrics: Precision, Recall, F1-Score, ROC-AUC

---

## 📊 Results

**Best Model: XGBoost**
- Precision (Fraud): 0.70
- Recall (Fraud): 0.75
- F1-Score (Fraud): 0.73
- ROC-AUC: 0.9683

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn imblearn xgboost scipy
