# -Healthcare-Provider-Fraud-Detection
This project aims to develop a robust machine learning solution to identify and predict potentially fraudulent healthcare providers based on their claims and beneficiary data. By leveraging historical claims information, the goal is to uncover patterns indicative of fraudulent behavior, thereby assisting insurance companies in mitigating financial losses and ensuring fair healthcare costs for beneficiaries.

2. Problem Statement
Healthcare provider fraud is a pervasive and costly issue within the insurance industry, leading to exponential increases in healthcare spending. This organized crime involves various parties collaborating to submit fraudulent claims, often through deceptive practices such as billing for unprovided services, duplicate claims, misrepresentation of services, or upcoding. The core challenge is to accurately predict which providers are likely to be fraudulent and to understand the underlying characteristics that distinguish them.

3. Dataset Overview
The project utilizes several interconnected datasets:

Train_Beneficiarydata.csv: Contains demographic and health condition details (KYC) for beneficiaries, including age, gender, race, and presence of various chronic conditions.

Train_Inpatientdata.csv: Provides detailed information on claims filed for patients admitted to hospitals, including claim amounts, physician IDs, admission/discharge dates, and diagnosis/procedure codes.

Train_Outpatientdata.csv: Contains details on claims for patients who visited hospitals but were not admitted, with similar claim and physician information as the inpatient data.

Train.csv: A mapping file that identifies providers as either PotentialFraud ('Yes') or No ('No'), serving as the target variable for the predictive model.

4. Solution Approach and Pipeline
The solution follows a structured machine learning pipeline:

Data Management:

Loading all raw CSV datasets into Pandas DataFrames.

Converting date columns (DOB, DOD, ClaimStartDt, ClaimEndDt, AdmissionDt, DischargeDt) to datetime objects.

Handling missing values: DOD missing values were addressed by calculating age at a reference date. DeductibleAmtPaid missing values were imputed with 0. Sparse physician ID and diagnosis/procedure code columns were handled during feature engineering.

Mapping categorical indicators (RenalDiseaseIndicator, ChronicCond_X) to numerical (0/1).

Feature Engineering:

Claim-Level Features: Calculated LengthOfStay (for inpatient), ClaimDuration, NumDiagnosisCodes, NumProcedureCodes, and binary flags for HasAttendingPhysician, HasOperatingPhysician, HasOtherPhysician for each claim.

Data Merging: Beneficiary data was merged with both inpatient and outpatient claims to enrich claim records with beneficiary-specific attributes.

Provider-Level Aggregation: Key features were aggregated from claim and beneficiary data to the Provider level. This included sums, means, min/max, and standard deviations of reimbursement and deductible amounts; counts of claims, unique beneficiaries, and unique physicians; average claim durations; and prevalence of chronic conditions among a provider's patients.

Combined Features: Cross-claim-type features like TotalClaims, TotalReimbursement, AvgReimbursementPerClaim, and various ratios were created.

The final dataset for modeling contained zero missing values after comprehensive handling.

Data Preparation for Modeling:

Separated features (X) from the target variable (y - PotentialFraud).

Identified numerical and categorical features for appropriate preprocessing.

Applied StandardScaler to numerical features and OneHotEncoder to categorical features using ColumnTransformer.

Split the data into training and validation sets (80/20 split) using stratify=y to maintain class distribution.

Addressing Class Imbalance:

The target variable PotentialFraud is highly imbalanced (approx. 90% 'No' vs. 10% 'Yes').

SMOTE (Synthetic Minority Oversampling Technique) was applied to the training data only to oversample the minority class and balance the dataset, preventing model bias towards the majority class.

Modelling and Hyperparameter Tuning:

Evaluated two classification algorithms: Logistic Regression and XGBoost.

RandomizedSearchCV with 5-fold cross-validation was used for hyperparameter tuning for each model.

The scoring metric for tuning was roc_auc, which is robust for imbalanced datasets.

Model Evaluation:

Models were evaluated on the unseen validation set using metrics critical for fraud detection:

Precision: Minimizes false positives (incorrectly flagging legitimate providers).

Recall: Minimizes false negatives (missing actual fraudulent providers).

F1-Score: Harmonic mean of precision and recall.

ROC-AUC: Overall discriminatory power.

Confusion Matrix: Detailed breakdown of predictions.

5. Key Findings and Model Performance
Both Logistic Regression and XGBoost models demonstrated strong capabilities in identifying potentially fraudulent providers.

Logistic Regression Performance:
Precision (Fraud): 0.49

Recall (Fraud): 0.89

F1-Score (Fraud): 0.63

ROC-AUC Score: 0.9686

Confusion Matrix (Validation Set):

Actual No Fraud / Predicted No Fraud: 887

Actual No Fraud / Predicted Fraud: 94 (False Positives)

Actual Fraud / Predicted No Fraud: 11 (False Negatives)

Actual Fraud / Predicted Fraud: 90 (True Positives)

XGBoost Performance:
Precision (Fraud): 0.70

Recall (Fraud): 0.75

F1-Score (Fraud): 0.73

ROC-AUC Score: 0.9683

Confusion Matrix (Validation Set):

Actual No Fraud / Predicted No Fraud: 949

Actual No Fraud / Predicted Fraud: 32 (False Positives)

Actual Fraud / Predicted No Fraud: 25 (False Negatives)

Actual Fraud / Predicted Fraud: 76 (True Positives)

Comparative Analysis:
While both models achieved high ROC-AUC scores, XGBoost significantly outperformed Logistic Regression in terms of Precision for the fraud class (0.70 vs. 0.49). This indicates that XGBoost is more effective at reducing false positives, which is crucial for minimizing the burden of investigating legitimate providers. Logistic Regression had a slightly higher Recall, but XGBoost offered a better balance between Precision and Recall (higher F1-Score). XGBoost is generally preferred for its balanced performance and lower false positive rate.

6. Business Recommendations
Proactive Risk Scoring: Implement the trained XGBoost model to assign fraud risk scores to providers, enabling early flagging of suspicious entities.

Targeted Investigations: Prioritize investigation efforts on providers with high fraud risk scores, optimizing resource allocation.

Continuous Monitoring: Regularly retrain the model with new data to adapt to evolving fraud patterns and maintain its effectiveness.

Feature Importance Analysis: Utilize the model's feature importance insights to understand specific fraudulent behaviors and inform fraud prevention strategies.

7. Setup and How to Run
To run this project, you will need:

Python 3.x

Required Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, imblearn, xgboost, scipy.
You can install them via pip:

pip install pandas numpy scikit-learn matplotlib seaborn imblearn xgboost scipy

Data Files: Ensure the following CSV files are in the same directory as the Python script:

Train_Beneficiarydata-1542865627584.csv

Train_Inpatientdata-1542865627584.csv

Train_Outpatientdata-1542865627584.csv

Train-1542865627584.csv

The main script (full_fraud_prediction_pipeline.py if you save the combined code) can be executed directly:

python full_fraud_prediction_pipeline.py

The script will perform all data loading, feature engineering, model training, and evaluation steps, printing results to the console.

8. Future Work and Improvements
Test Data Prediction: Apply the best-performing model to the unseen test dataset to generate final predictions for submission.

Advanced Feature Engineering: Explore more sophisticated features, such as network analysis of provider-beneficiary relationships, or temporal features to capture changes in billing patterns over time.

Deep Dive into Diagnosis/Procedure Codes: Develop more granular features from diagnosis and procedure codes, potentially using NLP techniques or expert-defined categories to identify "ambiguous" or high-cost codes.

Cost-Sensitive Learning: Investigate cost-sensitive learning approaches to explicitly account for the different costs associated with false positives and false negatives in fraud detection.

Deployment: Consider how the model could be deployed in a real-world claims processing system for continuous, automated fraud detection.
