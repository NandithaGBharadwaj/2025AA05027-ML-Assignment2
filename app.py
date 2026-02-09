import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# Page Configuration
st.set_page_config(
    page_title="ML Classification App",
    layout="wide"
)

# Gentle Left Alignment
st.markdown(
    """
    <style>
    .block-container {
        padding-left: 10rem;
        padding-right: 2rem;
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Machine Learning Classification Models")

# Student Details
st.markdown(
    """
    **Name:** Nanditha G Bharadwaj  
    **BITS ID:** 2025AA05027  
    **BITS Email ID:** 2025aa05027@wilp.bits-pilani.ac.in
    """
)

st.write("Upload test data, select a model, and view evaluation results.")

# Load Models & Preprocessors
models = {
    "Logistic Regression": joblib.load("model/Logistic_Regression.pkl"),
    "Decision Tree": joblib.load("model/Decision_Tree.pkl"),
    "KNN": joblib.load("model/KNN.pkl"),
    "Naive Bayes": joblib.load("model/Naive_Bayes.pkl"),
    "Random Forest": joblib.load("model/Random_Forest.pkl"),
    "XGBoost": joblib.load("model/XGBoost.pkl")
}

scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

# UI Components (Upload | Gap | Select | Gap | Download)
col1, gap1, col2, gap2, col3, _ = st.columns([2, 1, 2, 1, 2, 4])

with col1:
    st.subheader("Upload Test Dataset (CSV)")
    uploaded_file = st.file_uploader(
        label="",
        type=["csv"]
    )

with col2:
    st.subheader("Select Classification Model")
    model_name = st.selectbox(
        label="",
        options=list(models.keys())
    )

with col3:
    st.subheader("Download Test Dataset")

    test_csv_url = (
        "https://raw.githubusercontent.com/"
        "NandithaGBharadwaj/2025AA05027-ML-Assignment2/main/model/test.csv"
    )

    response = requests.get(test_csv_url)

    st.download_button(
        label="Download test.csv",
        data=response.content,
        file_name="test.csv",
        mime="text/csv"
    )

# Prediction and Metrics
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "HeartDisease" not in data.columns:
        st.error("CSV must contain 'HeartDisease' column.")
        st.stop()

    categorical_cols = [
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope"
    ]

    for col in categorical_cols:
        if col in data.columns and data[col].dtype == "object":
            encoder = label_encoders[col]
            mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
            data[col] = data[col].map(mapping)

    if data[categorical_cols].isnull().any().any():
        st.error(
            "Encoding mismatch detected.\n"
            "Ensure test.csv was generated from training script."
        )
        st.stop()

    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]

    X_scaled = scaler.transform(X)
    model = models[model_name]

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Metrics & Confusion Matrix (with 1-column gap)
    metrics_col, gap3, cm_col, _ = st.columns([2.5, 1, 2.5, 4])

    with metrics_col:
        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(y, y_pred))
        st.write("AUC:", roc_auc_score(y, y_prob))
        st.write("Precision:", precision_score(y, y_pred))
        st.write("Recall:", recall_score(y, y_pred))
        st.write("F1 Score:", f1_score(y, y_pred))
        st.write("MCC:", matthews_corrcoef(y, y_pred))

    with cm_col:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
