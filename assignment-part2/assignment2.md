## Task 1: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
## Task 2: Load the Dataset

df = pd.read_csv("cleaned_data.csv")

## Display first 5 rows
print(df.head())

## Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())
## Task 3: Define X, y_reg and y_clf

## Regression 
y_reg = df["Sales"]

## Binary Classification Label
y_clf = (df["Sales"] > df["Sales"].median()).astype(int)

## Feature Matrix
X = df.drop(columns=["Sales"])

print("Feature Matrix Shape :", X.shape)
print("Regression Label Shape :", y_reg.shape)
print("Classification Label Shape :", y_clf.shape)
## Task 4.1: Convert Date Column

X["Order Date"] = pd.to_datetime(X["Order Date"])

X["Order_Year"] = X["Order Date"].dt.year
X["Order_Month"] = X["Order Date"].dt.month
X["Order_Day"] = X["Order Date"].dt.day

## Drop original date column
X = X.drop(columns=["Order Date"])
## Task 4.2: Remove Unique Identifier Columns

X = X.drop(columns=["Row ID", "Order ID"])
## Task 4.3: Check Missing Values

print(X.isnull().sum())
X["Profit"] = X["Profit"].fillna(X["Profit"].median())
## Task 4.4: One-Hot Encoding

categorical_columns = X.select_dtypes(include=["object"]).columns

print("Categorical Columns:")
print(categorical_columns)

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("\nShape after Encoding:")
print(X.shape)
## Task 5: Train-Test Split

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X,
    y_reg,
    test_size=0.20,
    random_state=42
)

## Split classification target using same random state
_, _, y_clf_train, y_clf_test = train_test_split(
    X,
    y_clf,
    test_size=0.20,
    random_state=42
)

print("Training Features :", X_train.shape)
print("Testing Features :", X_test.shape)
## Task 6: Feature Scaling

scaler = StandardScaler()

## Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

## Transform testing data
X_test_scaled = scaler.transform(X_test)

print("Training Data Shape :", X_train_scaled.shape)
print("Testing Data Shape :", X_test_scaled.shape)
## Task 7: Train Linear Regression Model

from sklearn.linear_model import LinearRegression

## Create the model
linear_model = LinearRegression()

## Train the model
linear_model.fit(X_train_scaled, y_reg_train)

print("Linear Regression model trained successfully.")
## Task 8: Evaluate Linear Regression Model

from sklearn.metrics import mean_squared_error, r2_score

## Predict Sales
y_pred_reg = linear_model.predict(X_test_scaled)

## Evaluation Metrics
mse = mean_squared_error(y_reg_test, y_pred_reg)
r2 = r2_score(y_reg_test, y_pred_reg)

print("Linear Regression Results")
print("-------------------------")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
## Task 9: Feature Coefficients

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})

## Absolute values
coefficients["Absolute Coefficient"] = coefficients["Coefficient"].abs()

## Sort
coefficients = coefficients.sort_values(
    by="Absolute Coefficient",
    ascending=False
)
## Task 10: Ridge Regression

from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(
    X_train_scaled,
    y_reg_train
)

ridge_predictions = ridge_model.predict(
    X_test_scaled
)

ridge_mse = mean_squared_error(
    y_reg_test,
    ridge_predictions
)

ridge_r2 = r2_score(
    y_reg_test,
    ridge_predictions
)

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression"],
    "MSE": [mse, ridge_mse],
    "R² Score": [r2, ridge_r2]
})

print(comparison)
## Task 11: Check Class Distribution

print("Class Distribution")

print(y_clf_train.value_counts())

print("\nPercentage Distribution")

print(y_clf_train.value_counts(normalize=True) * 100)
## Task 12: Train Logistic Regression Model

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
## Create the model
model.fit(
    X_train_scaled,
    y_clf_train
)

print("Logistic Regression model trained successfully.")
## Task 13: Logistic Regression Predictions

## Predict class labels
y_pred = model.predict(X_test_scaled)

## Predict probabilities
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Predictions completed successfully.")
## Task 14: Evaluate Logistic Regression

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

## Confusion Matrix
cm = confusion_matrix(y_clf_test, y_pred)

print("Confusion Matrix")
print(cm)

## Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

## Performance Metrics
accuracy = accuracy_score(y_clf_test, y_pred)
precision = precision_score(y_clf_test, y_pred)
recall = recall_score(y_clf_test, y_pred)
f1 = f1_score(y_clf_test, y_pred)

print("\nAccuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)

print("\nClassification Report")
print(classification_report(y_clf_test, y_pred))
## Task 15: ROC Curve and AUC

from sklearn.metrics import roc_curve, roc_auc_score

## ROC values
fpr, tpr, thresholds = roc_curve(y_clf_test, y_prob)

## AUC Score
auc = roc_auc_score(y_clf_test, y_prob)

print("AUC Score :", auc)

## Plot ROC Curve
plt.figure(figsize=(7,6))

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.show()
## Task 16: Threshold Sensitivity

thresholds = [0.30,0.40,0.50,0.60,0.70]

results=[]

for threshold in thresholds:

    predictions = (y_prob >= threshold).astype(int)

    precision = precision_score(y_clf_test, predictions)
    recall = recall_score(y_clf_test, predictions)
    f1 = f1_score(y_clf_test, predictions)

    results.append([
        threshold,
        precision,
        recall,
        f1
    ])

threshold_table = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print(threshold_table)
## Task 17: Logistic Regression (C = 0.01)

strong_model = LogisticRegression(
    C=0.01,
    max_iter=1000,
    random_state=42
)

strong_model.fit(
    X_train_scaled,
    y_clf_train
)

## Predictions
strong_pred = strong_model.predict(X_test_scaled)

strong_prob = strong_model.predict_proba(
    X_test_scaled
)[:,1]

comparison = pd.DataFrame({

    "Model":[
        "C = 1.0",
        "C = 0.01"
    ],

    "Precision":[
        precision_score(y_clf_test,y_pred),
        precision_score(y_clf_test,strong_pred)
    ],

    "Recall":[
        recall_score(y_clf_test,y_pred),
        recall_score(y_clf_test,strong_pred)
    ],

    "AUC":[
        roc_auc_score(y_clf_test,y_prob),
        roc_auc_score(y_clf_test,strong_prob)
    ]

})

print(comparison)
## Task 18: Bootstrap Confidence Interval

auc_difference=[]

for i in range(500):

    sample_index = np.random.choice(
        len(y_clf_test),
        size=len(y_clf_test),
        replace=True
    )

    y_bootstrap = y_clf_test.iloc[sample_index]

    prob1 = y_prob[sample_index]

    prob2 = strong_prob[sample_index]

    auc1 = roc_auc_score(
        y_bootstrap,
        prob1
    )

    auc2 = roc_auc_score(
        y_bootstrap,
        prob2
    )

    auc_difference.append(
        auc1 - auc2
    )

mean_difference = np.mean(auc_difference)

lower = np.percentile(
    auc_difference,
    2.5
)

upper = np.percentile(
    auc_difference,
    97.5
)

print("Mean AUC Difference :", mean_difference)
print("95% Confidence Interval")
print("Lower Bound :", lower)
print("Upper Bound :", upper)

if lower > 0 or upper < 0:
    print("\nThe confidence interval excludes zero.")
    print("The C=1.0 model consistently outperforms the C=0.01 model.")
else:
    print("\nThe confidence interval includes zero.")
    print("The performance difference may not be statistically significant.")
