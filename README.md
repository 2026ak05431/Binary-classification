
# Bank Marketing Subscription Prediction

## 1. Problem Statement

The objective of this project is to predict whether a customer
will subscribe to a bank term deposit based on demographic,
financial and previous marketing campaign information.

This is a supervised binary classification problem.

## 2. Dataset Description

Dataset: Bank Marketing Dataset

Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/222/bank+marketing

- Total instances: 45,211
- Original input features: 16
- Features used: 15
- Target: y
- Classes: Yes and No
- Classification: Binary

The duration feature was excluded to prevent
post-call information leakage.

## 3. GitHub Repository Link

[Insert GitHub Repository URL]

## 4. Models Used

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

### Evaluation Metrics

The models were evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient

### Model Comparison

[Insert the actual model_comparison.csv results]

### Model Performance Observations

Logistic Regression:
Evaluate its ability to identify the relationship
between customer features and subscription.

Decision Tree:
Evaluate its ability to capture nonlinear relationships
and its sensitivity to overfitting.

KNN:
Evaluate its performance based on neighborhood-based
classification.

Naive Bayes:
Evaluate its performance using probabilistic
classification assumptions.

Support Vector Machine:
Evaluate its ability to separate customer classes
using a nonlinear decision boundary.

### Overall Winner

[Insert the best-performing model based on actual
evaluation metrics.]

## 5. Streamlit Application

[Insert deployed Streamlit URL]

The application supports:
- CSV test dataset upload
- Model selection
- Evaluation metrics
- Confusion matrix
- Classification report
- Comparison of all five models

## 6. Conclusion

This project implements and compares five machine
learning classification algorithms for predicting
bank term deposit subscriptions.

The models are evaluated using multiple performance
metrics to account for the class imbalance in the dataset.
