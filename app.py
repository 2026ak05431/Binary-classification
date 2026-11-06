
import streamlit as st
import pandas as pd
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Bank Marketing ML Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("Bank Marketing Classification Dashboard")

st.write(
    "Predict whether a customer will subscribe to a bank term deposit "
    "using five machine learning classification algorithms."
)

models = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if "y" not in data.columns:
        st.error("Uploaded CSV must contain target column 'y'.")
        st.stop()

    X = data.drop(columns=["y"])
    y = data["y"].replace({
        "no": 0,
        "yes": 1,
        0: 0,
        1: 1
    })

    y = pd.to_numeric(y, errors="coerce")

    if y.isnull().any():
        st.error("Target column must contain only 'yes' or 'no'.")
        st.stop()

    if len(data) == 0 or y.nunique() != 2:
        st.error("Test dataset must contain both target classes.")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    results = {}
    loaded_models = {}

    for name, filename in models.items():

        path = os.path.join("models", filename)

        if not os.path.exists(path):
            st.error(f"Missing model: {filename}")
            st.stop()

        model = joblib.load(path)

        try:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            results[name] = {
                "Accuracy": accuracy_score(y, predictions),
                "AUC": roc_auc_score(y, probabilities),
                "Precision": precision_score(
                    y, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y, predictions, zero_division=0
                ),
                "F1": f1_score(
                    y, predictions, zero_division=0
                ),
                "MCC": matthews_corrcoef(y, predictions)
            }

            loaded_models[name] = (
                model, predictions
            )

        except Exception as error:
            st.error(f"{name} evaluation failed: {error}")
            st.stop()

    st.subheader("Model Comparison")

    comparison = pd.DataFrame(results).T
    st.dataframe(
        comparison.style.format("{:.4f}"),
        use_container_width=True
    )

    st.subheader("Select Model")

    selected = st.selectbox(
        "Choose classifier",
        list(models.keys())
    )

    model, predictions = loaded_models[selected]
    selected_metrics = results[selected]

    st.subheader(f"{selected} - Evaluation Metrics")

    columns = st.columns(3)

    for index, (metric, value) in enumerate(
        selected_metrics.items()
    ):
        columns[index % 3].metric(
            metric, f"{value:.4f}"
        )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y,
        predictions,
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).T.style.format("{:.4f}"),
        use_container_width=True
    )

else:
    st.info("Upload test_data.csv to evaluate all five models.")
