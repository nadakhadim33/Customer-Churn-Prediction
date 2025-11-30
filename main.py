# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# 2. Load Data
data = pd.read_csv(r"C:\Users\nada\OneDrive\Desktop\Customer_Churn_Prediction\Bank Customer Churn Prediction.csv")

# 3. Data Exploration
print(data.head())
print(data.info())
print(data.describe())
print(data['churn'].value_counts())

# 4. Data Cleaning
print(data.isnull().sum())
data = data.drop(['customer_id'], axis=1)

# Encode categorical variables
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['country'] = le.fit_transform(data['country'])

# Features and Target
X = data.drop('churn', axis=1)
y = data['churn']

# 5. Handle Imbalanced Data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 7. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Model Training
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 9. Model Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_log))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix Visualization
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=axes[1], cmap='Greens')
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 10. Feature Importance (Random Forest)
importances = rf_model.feature_importances_
features = X.columns
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)

# 11. Save Model
if not os.path.exists('outputs'):
    os.makedirs('outputs')

joblib.dump(rf_model, 'outputs/random_forest_churn_model.pkl')
joblib.dump(scaler, 'outputs/scaler.pkl')
