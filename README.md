# Machine Learning Assignment 2 – Classification Models
The objective of this assignment is to implement multiple machine learning
classification models on a real-world dataset, evaluate their performance using
standard metrics, and deploy the trained models through an interactive Streamlit
web application.

------------------------------------------------------------------

## a. PROBLEM STATEMENT
# Heart Disease Classification using Machine Learning


Heart disease is one of the leading causes of death worldwide. Early and accurate prediction of heart disease can help doctors take preventive measures and provide timely treatment.

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a patient has heart disease based on clinical and demographic data.

The target variable is HeartDisease:
0 -> No heart disease
1 -> Presence of heart disease

## b. Dataset Description
The Heart Disease dataset has been used for this assignment. It contains medical
attributes such as age, cholesterol, blood pressure, and heart rate to predict the
presence of heart disease.

- Dataset Source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- Number of instances: 918
- Number of features: 12
- Target variable: Binary classification (presence or absence of heart disease)


FEATURES IN THE DATASET:

- **Age**: Age of the patient  
- **Sex**: Gender of the patient  
- **ChestPainType**: Type of chest pain  
- **RestingBP**: Resting blood pressure  
- **Cholesterol**: Serum cholesterol level  
- **FastingBS**: Fasting blood sugar  
- **RestingECG**: Resting electrocardiographic results  
- **MaxHR**: Maximum heart rate achieved  
- **ExerciseAngina**: Exercise-induced angina  
- **Oldpeak**: ST depression induced by exercise  
- **ST_Slope**: Slope of peak exercise ST segment  
- **HeartDisease**: Target variable (0 or 1)

PREPROCESSING STEPS:
- Categorical features encoded using Label Encoding
- Numerical features scaled using StandardScaler
- Data split into 80% training and 20% testing using stratified sampling

------------------------------------------------------------------
## c. Models Used and Evaluation Metrics

The following machine learning models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Evaluation Metrics Used:
- Accuracy
- AUC (Area Under ROC Curve)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


MODEL COMPARISON TABLE

ML Model Name        | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC
---------------------|----------|---------|-----------|---------|----------|---------
Logistic Regression  | 0.8696   | 0.8971  | 0.8482    | 0.9314  | 0.8879   | 0.7374
Decision Tree        | 0.7880   | 0.7861  | 0.8119    | 0.8039  | 0.8079   | 0.5716
kNN                  | 0.8913   | 0.9192  | 0.8942    | 0.9118  | 0.9029   | 0.7797
Naive Bayes          | 0.8913   | 0.9280  | 0.8942    | 0.9118  | 0.9029   | 0.7797
Random Forest        | 0.8750   | 0.9229  | 0.8762    | 0.9020  | 0.8889   | 0.7465
XGBoost              | 0.8696   | 0.9230  | 0.8980    | 0.8627  | 0.8800   | 0.7380


## Model Performance Observations

| ML Model Name        | Observation about Model Performance |
|----------------------|-------------------------------------|
| Logistic Regression  | Achieved high recall and good overall performance. Effective for this dataset but limited by linear boundaries. |
| Decision Tree        | Lower performance compared to other models. Likely overfitting and poor generalization. |
| kNN                  | One of the best-performing models. High accuracy, F1 score, and MCC after feature scaling. |
| Naive Bayes          | Strong and consistent performance. Independence assumption worked well for this dataset. |
| Random Forest        | Improved performance over Decision Tree. Ensemble method reduced overfitting and improved stability. |
| XGBoost              | High precision and strong overall metrics. Gradient boosting captured complex patterns effectively. |

------------------------------------------------------------------

CONCLUSION

The results show that kNN and Naive Bayes achieved the highest overall performance, while Random Forest and XGBoost provided
robust and stable results due to their ensemble nature.

Ensemble and probabilistic models proved to be well-suited for heart disease classification on the given dataset.


------------------------------------------------------------------


# 2025AA05027-ML-Assignment2
>>>>>>> 016c2cff917764566c25b9608e4551580bb62690
