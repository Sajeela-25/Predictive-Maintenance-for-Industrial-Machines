import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import joblib
#1. Load Dataset
df = pd.read_csv("ai4i2020.csv")

#Rename columns for simplicity
df.rename(columns={
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "failure"
}, inplace=True)


#2. Handle Missing Values
#Dataset has no missing values, but handled for safety
df.fillna(df.median(numeric_only=True), inplace=True)

#3. Binning / Discretization
df["rpm_bin"] = pd.cut(df["rpm"], bins=3, labels=["Low", "Medium", "High"])
df["temp_bin"] = pd.cut(df["process_temp"], bins=3, labels=["Low", "Medium", "High"])

#4. Outlier Detection (IQR)
def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    return column.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

for col in ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]:
    df[col] = remove_outliers(df[col])

#5. Feature Selection
features = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
X = df[features]
y = df["failure"]

#6. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#7. Dimensionality Reduction (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#9. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#10. Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Model Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC-AUC:", roc_auc)

#11. Visualizations
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Feature Importance
importances = model.feature_importances_
plt.bar(features, importances)
plt.title("Feature Importance")
plt.show()

#PCA Visualization
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm")
plt.title("PCA Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#12. Save Model & Scaler
joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
