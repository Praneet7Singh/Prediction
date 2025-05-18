import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data=pd.read_csv('Cardiovascular_Disease_Dataset.csv')

median_cholesterol = data[data['serumcholestrol'] > 0]['serumcholestrol'].median()
data.loc[data['serumcholestrol'] == 0, 'serumcholestrol'] = median_cholesterol


X = data.drop(['target','patientid'], axis=1)
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump({'model': rf_model, 'scaler': scaler}, f)