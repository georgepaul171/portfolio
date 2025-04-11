import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train_model import train
from src.evaluate import evaluate
from src.shap_explainer import explain
st.title("Customer Churn Prediction")

file = st.file_uploader("Upload Telco Churn CSV", type=["csv"])
model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression", "XGBoost"])

if file:
    df = load_data(file)
    st.write("Sample Data", df.head())

    X, y = preprocess_data(df)
    model, X_test, y_test, cv_score = train(X, y, model_choice)
    st.success(f"{model_choice} trained successfully with CV F1 Score: {cv_score:.2f}")

    metrics = evaluate(model, X_test, y_test)
    st.subheader("Model Evaluation")
    st.json(metrics)

    st.subheader("SHAP Feature Importance")
    shap_img_path = explain(model, X)
    st.image(shap_img_path)
