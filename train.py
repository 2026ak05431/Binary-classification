
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Load dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

# Exclude duration to avoid post-call information leakage
df.drop(columns=["duration"], inplace=True)

X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})

# 2. Identify feature types
numerical = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(include="object").columns.tolist()

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore",
                          sparse_output=False), categorical)
])

# 5. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12, random_state=42,
        class_weight="balanced"
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(
        kernel="rbf", probability=True,
        class_weight="balanced", random_state=42
    )
}

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

results = []

# 6. Train and evaluate
for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions)
    }

    results.append(metrics)

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipeline, f"models/{filename}")

    print(metrics)

# 7. Save comparison results
comparison = pd.DataFrame(results)
comparison.to_csv("outputs/model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

# 8. Save test dataset for Streamlit
test_data = X_test.copy()
test_data["y"] = y_test.map({
    0: "no",
    1: "yes"
})

test_data.to_csv("test_data.csv", index=False)

print("\n===================================")
print("ALL FIVE MODELS TRAINED SUCCESSFULLY")
print("Models saved in models/")
print("Comparison saved in outputs/")
print("Test data saved in test_data.csv")
print("===================================")
