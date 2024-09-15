import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

# Step 1: Load the data
application_train = pd.read_csv('application_train.csv')

# Step 2: Handle missing values
application_train.fillna(application_train.median(numeric_only=True), inplace=True)
application_train.fillna(application_train.mode().iloc[0], inplace=True)

# Step 3: Encode categorical variables
categorical_cols = application_train.select_dtypes(include=[object]).columns
le = LabelEncoder()
for col in categorical_cols:
    if application_train[col].nunique() == 2:
        application_train[col] = le.fit_transform(application_train[col])
    else:
        application_train = pd.get_dummies(application_train, columns=[col], drop_first=True)

# Step 4: Feature Engineering
application_train['CREDIT_INCOME_PERCENT'] = application_train['AMT_CREDIT'] / application_train['AMT_INCOME_TOTAL']
application_train['ANNUITY_INCOME_PERCENT'] = application_train['AMT_ANNUITY'] / application_train['AMT_INCOME_TOTAL']
application_train['CREDIT_TERM'] = application_train['AMT_ANNUITY'] / application_train['AMT_CREDIT']
application_train['DAYS_EMPLOYED_PERCENT'] = application_train['DAYS_EMPLOYED'] / application_train['DAYS_BIRTH']

# Step 5: Scaling numerical features
numeric_features = application_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
application_train[numeric_features] = scaler.fit_transform(application_train[numeric_features])

# Step 6: Prepare data for modeling
X = application_train.drop(columns=['TARGET', 'SK_ID_CURR'])

# Ensure target is binary (if required)
y = application_train['TARGET'].apply(lambda x: 1 if x >= 0.5 else 0)

# Step 7: Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train a simple Logistic Regression model
model = LogisticRegression(C=1, solver='liblinear', max_iter=200, random_state=42)
model.fit(X_train, y_train)

# Step 9: Save the model and scaler
joblib.dump(model, 'optimized_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Step 10: Print ROC-AUC score
y_pred_prob = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_prob)
print(f"Model ROC-AUC Score: {roc_auc:.4f}")
