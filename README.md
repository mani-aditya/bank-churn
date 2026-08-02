# Bank Customer Churn Prediction

A machine learning pipeline developed in Python to predict customer behavior (churn/conversion) using the `bank-full.csv` dataset. This project compares the performance of Logistic Regression and K-Nearest Neighbors (KNN) classifiers on a class-imbalanced dataset.

## Features

* **Data Preprocessing:** Handles categorical variables using `LabelEncoder` and standardizes numerical features with `StandardScaler`.
* **Class Imbalance Handling:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class, ensuring the models train on a perfectly balanced dataset.
* **Model Training & Evaluation:** Trains both Logistic Regression and KNN (k=5) models.
* **Performance Metrics:** Evaluates models using Accuracy, Precision, Classification Reports, and Confusion Matrices.

## Models Compared

* **Logistic Regression:** Used as a baseline model for binary classification.
* **K-Nearest Neighbors (KNN):** A non-parametric instance-based learning algorithm.
